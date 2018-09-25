
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer as KerasLayer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging

from tensorflow.python.layers import base


class MovingFreeBatchNormalizationKeras(KerasLayer):

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 mean_initializer='zeros',
                 variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 fused=None,
                 trainable=True,
                 virtual_batch_size=None,
                 adjustment=None,
                 name=None,
                 **kwargs):
        super(MovingFreeBatchNormalizationKeras, self).__init__(
            name=name, trainable=trainable, **kwargs)
        if isinstance(axis, list):
            self.axis = axis[:]
        else:
            self.axis = axis
        self.epsilon = epsilon
        self.momentum = momentum
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.mean_initializer = initializers.get(mean_initializer)
        self.variance_initializer = initializers.get(variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.virtual_batch_size = virtual_batch_size
        self.adjustment = adjustment
        if fused is None:
            fused = True
        self.supports_masking = True

        self.fused = fused
        self._bessels_correction_test_only = True

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        # Convert axis to list and resolve negatives
        if isinstance(self.axis, int):
            self.axis = [self.axis]

        if not isinstance(self.axis, list):
            raise TypeError('axis must be int or list, type given: %s'
                            % type(self.axis))

        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
            if x < 0 or x >= ndims:
                raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
            raise ValueError('Duplicate axis: %s' % self.axis)

        if self.virtual_batch_size is not None:
            if self.virtual_batch_size <= 0:
                raise ValueError('virtual_batch_size must be a positive integer that '
                                 'divides the true batch size of the input Tensor')
            # If using virtual batches, the first dimension must be the batch
            # dimension and cannot be the batch norm axis
            if 0 in self.axis:
                raise ValueError('When using virtual_batch_size, the batch dimension '
                                 'must be 0 and thus axis cannot include 0')
            if self.adjustment is not None:
                raise ValueError('When using virtual_batch_size, adjustment cannot '
                                 'be specified')

        if self.fused:
            self.fused = (ndims == 4 and
                          self.axis in [[1], [3]] and
                          self.virtual_batch_size is None and
                          self.adjustment is None)
        if self.fused:
            if self.axis == [1]:
                self._data_format = 'NCHW'
            elif self.axis == [3]:
                self._data_format = 'NHWC'
            else:
                raise ValueError('Unsupported axis, fused batch norm only supports '
                                 'axis == [1] or axis == [3]')

        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            param_dtype = dtypes.float32
        else:
            param_dtype = self.dtype or dtypes.float32

        axis_to_dim = {x: input_shape[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                                 input_shape)
        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        if len(axis_to_dim) == 1 and self.virtual_batch_size is None:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis dims
            param_shape = [axis_to_dim[i] if i in axis_to_dim
                           else 1 for i in range(ndims)]
            if self.virtual_batch_size is not None:
                # When using virtual batches, add an extra dim at index 1
                param_shape.insert(1, 1)
                for idx, x in enumerate(self.axis):
                    self.axis[idx] = x + 1      # Account for added dimension

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True)
        else:
            self.gamma = None
            if self.fused:
                self._gamma_const = array_ops.constant(
                    1.0, dtype=param_dtype, shape=param_shape)

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True)
        else:
            self.beta = None
            if self.fused:
                self._beta_const = array_ops.constant(
                    0.0, dtype=param_dtype, shape=param_shape)

        try:
            # Disable variable partitioning when creating the moving mean and variance
            if hasattr(self, '_scope') and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None

            # internal statistics fitted during the training
            self.moving_mean = self.add_weight(
                name='moving_mean',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.mean_initializer,
                synchronization=variable_scope.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=variable_scope.VariableAggregation.MEAN)

            self.moving_variance = self.add_weight(
                name='moving_variance',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.variance_initializer,
                synchronization=variable_scope.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=variable_scope.VariableAggregation.MEAN)

            # internal statistics fitted during a pre-inference step
            self.mean = self.add_weight(
                name='mean',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.mean_initializer,
                synchronization=variable_scope.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=variable_scope.VariableAggregation.MEAN)

            self.variance = self.add_weight(
                name='variance',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.variance_initializer,
                synchronization=variable_scope.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=variable_scope.VariableAggregation.MEAN)

            self.n_updates = self.add_weight(
                name='n_updates',
                shape=[],
                dtype=param_dtype,
                initializer=initializers.get('zeros'),
                synchronization=variable_scope.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=variable_scope.VariableAggregation.MEAN)
        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def _update_statistics(self, variable, value, n_updates):
        with ops.name_scope(None, 'UpdateStatistics',
                            [variable, value, n_updates]) as scope:
            with ops.colocate_with(variable):
                stat = variable*n_updates + value
                stat /= n_updates + 1
                return state_ops.assign(variable, stat, name=scope)

    def _assign_moving_averages(self, variable, value, momentum):

        with ops.name_scope(None, 'AssignMovingAvg', [variable, value, momentum]) as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (variable - math_ops.cast(value, variable.dtype.base_dtype)) * decay
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def _fused_batch_norm(self, inputs, training, use_moving_statistics):
        """Returns the output of fused batch norm."""
        beta = self.beta if self.center else self._beta_const
        gamma = self.gamma if self.scale else self._gamma_const

        def _fused_batch_norm_training():
            return nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                epsilon=self.epsilon,
                data_format=self._data_format)

        mean = tf_utils.smart_cond(use_moving_statistics, lambda: self.moving_mean, lambda : self.mean)
        variance = tf_utils.smart_cond(use_moving_statistics, lambda: self.moving_variance, lambda : self.variance)

        def _fused_batch_norm_inference():
            return nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=mean,
                variance=variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format)

        output, mean, variance = tf_utils.smart_cond(training, _fused_batch_norm_training, _fused_batch_norm_inference)
        # if training == True: meand and variance returned are mean and vriance of the current batch
        # elif training == False: mean and variance return are (self.mean, self.variance) or
        #   (self.moving_mean, self.moving_variance) depending of the value of use_moving_statistics

        if not self._bessels_correction_test_only:
            # Remove Bessel's correction to be consistent with non-fused batch norm.
            # Note that the variance computed by fused batch norm is
            # with Bessel's correction.
            sample_size = math_ops.cast(
                array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
            factor = (sample_size - math_ops.cast(1.0, variance.dtype)) / sample_size
            variance *= factor

        training_value = tf_utils.constant_value(training)

        if training_value is None:
            momentum = tf_utils.smart_cond(training,
                                           lambda: self.momentum,
                                           lambda: 1.0)
        else:
            momentum = ops.convert_to_tensor(self.momentum)

        if training_value or training_value is None:

            # if training, first create operations which update self.mean and self.variance
            mean_update = self._update_statistics(self.mean, mean, self.n_updates)
            variance_update = self._update_statistics(self.variance, variance, self.n_updates)
            with ops.control_dependencies([mean_update, variance_update]):
                update_n_updates = state_ops.assign_add(self.n_updates, 1., )

            # add this combination of operatio to a specific collection
            ops.add_to_collection('UPDATE_BN_OPS', update_n_updates)

            # operations to reset bn statistics
            reset_mean = state_ops.assign(self.mean, array_ops.zeros_like(self.mean))
            reset_variance = state_ops.assign(self.variance, array_ops.zeros_like(self.variance))
            reset_n_updates = state_ops.assign(self.n_updates, 0.)
            ops.add_to_collection('RESET_BN_OPS', reset_mean)
            ops.add_to_collection('RESET_BN_OPS', reset_variance)
            ops.add_to_collection('RESET_BN_OPS', reset_n_updates)

            # update moving averages and add operations to tf.GraphKeys.UPDATE_OPS
            # these operation must be run when optimizing the network
            moving_mean_update = self._assign_moving_averages(self.moving_mean, mean, momentum)
            moving_variance_update = self._assign_moving_averages(self.moving_variance, variance, momentum)
            self.add_update(moving_mean_update, inputs=True)
            self.add_update(moving_variance_update, inputs=True)


        return output

    def call(self, inputs, training=None, use_moving_statistics=True):
        original_training_value = training
        if training is None:
            training = K.learning_phase()

        in_eager_mode = context.executing_eagerly()
        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping the
            # Tensor and reusing the existing batch norm implementation
            original_shape = [-1] + inputs.shape.as_list()[1:]
            expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

            # Will cause errors if virtual_batch_size does not divide the batch size
            inputs = array_ops.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = array_ops.reshape(outputs, original_shape)
                return outputs

        if self.fused:
            outputs = self._fused_batch_norm(inputs, training=training, use_moving_statistics=use_moving_statistics)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not support
                # virtual batching
                outputs = undo_virtual_batching(outputs)
            if not context.executing_eagerly() and original_training_value is None:
                outputs._uses_learning_phase = True  # pylint: disable=protected-access
            return outputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.get_shape()
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]     # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape[self.axis[0]].value
        def _broadcast(v):
            if (v is not None and
                        len(v.get_shape()) != ndims and
                        reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        use_moving_statistics_value = tf_utils.constant_value(use_moving_statistics)

        if training_value is not False:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
                # Adjust only during training.
                adj_scale = tf_utils.smart_cond(training,
                                                lambda: adj_scale,
                                                lambda: array_ops.ones_like(adj_scale))
                adj_bias = tf_utils.smart_cond(training,
                                               lambda: adj_bias,
                                               lambda: array_ops.zeros_like(adj_bias))
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1

            # mean and variance of the current batch
            mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)
            
            mean = tf_utils.smart_cond(training,
                                       lambda: mean,
                                       lambda: tf_utils.smart_cond(use_moving_statistics,
                                                                   lambda : self.moving_mean,
                                                                   lambda : self.mean))
            variance = tf_utils.smart_cond(training,
                                           lambda: variance,
                                           lambda: tf_utils.smart_cond(use_moving_statistics,
                                                                       lambda : self.moving_variance,
                                                                       lambda : self.variance))

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and moving_variance
                # with each sub-batch. However, since the moving statistics are only
                # used during evaluation, it is more efficient to just update in one
                # step and should not make a significant difference in the result.
                new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
            else:
                new_mean, new_variance = mean, variance

            if self.renorm:
                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    new_mean, new_variance, training)
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
                d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                if in_eager_mode and not self.trainable:
                    return
                return self._assign_moving_average(var, value, self.momentum)

            moving_mean_update = tf_utils.smart_cond(
                training,
                lambda: _do_update(self.moving_mean, new_mean),
                lambda: self.moving_mean)
            moving_variance_update = tf_utils.smart_cond(
                training,
                lambda: _do_update(self.moving_variance, new_variance),
                lambda: self.moving_variance)

            if not context.executing_eagerly():
                self.add_update(moving_mean_update, inputs=True)
                self.add_update(moving_variance_update, inputs=True)

            mean_update = self._update_statistics(self.mean, mean, self.n_updates)
            variance_update = self._update_statistics(self.variance, variance, self.n_updates)

            with ops.control_dependencies([mean_update, variance_update]):
                # update n_updates only after updating self.mean and self.variance
                update_n_updates = state_ops.assign_add(self.n_updates, 1.)
                ops.add_to_collection('UPDATE_BN_OPS', update_n_updates)

            reset_mean = state_ops.assign(self.mean, array_ops.zeros_like(self.mean))
            reset_variance = state_ops.assign(self.variance, array_ops.zeros_like(self.variance))
            reset_n_updates = state_ops.assign(self.n_updates, 0.)
            ops.add_to_collection('RESET_OPS', reset_mean)
            ops.add_to_collection('RESET_OPS', reset_variance)
            ops.add_to_collection('RESET_OPS', reset_n_updates)

        else:
            # training == False
            mean = tf_utils.smart_cond(use_moving_statistics, lambda: self.moving_mean, lambda: self.mean)
            variance = tf_utils.smart_cond(use_moving_statistics, lambda: self.moving_variance, lambda: self.variance)

        mean = math_ops.cast(mean, inputs.dtype)
        variance = math_ops.cast(variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
        outputs = nn.batch_normalization(inputs,
                                         _broadcast(mean),
                                         _broadcast(variance),
                                         offset,
                                         scale,
                                         self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        if not context.executing_eagerly() and original_training_value is None:
            outputs._uses_learning_phase = True  # pylint: disable=protected-access
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        # Only add TensorFlow-specific parameters if they are set, so as to preserve
        # model compatibility with external Keras.
        if self.virtual_batch_size is not None:
            config['virtual_batch_size'] = self.virtual_batch_size
        # Note: adjustment is not serializable.
        if self.adjustment is not None:
            logging.warning('The `adjustment` function of this `BatchNormalization` '
                            'layer cannot be serialized and has been omitted from '
                            'the layer config. It will not be included when '
                            're-creating the layer from the saved config.')
        base_config = super(MovingFreeBatchNormalizationKeras, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MovingFreeBatchNormalization(MovingFreeBatchNormalizationKeras, base.Layer):

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer=init_ops.zeros_initializer(),
                 gamma_initializer=init_ops.ones_initializer(),
                 mean_initializer=init_ops.zeros_initializer(),
                 variance_initializer=init_ops.ones_initializer(),
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 fused=None,
                 trainable=True,
                 virtual_batch_size=None,
                 adjustment=None,
                 name=None,
                 **kwargs):
        super(MovingFreeBatchNormalization, self).__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            mean_initializer=mean_initializer,
            variance_initializer=variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            fused=fused,
            trainable=trainable,
            virtual_batch_size=virtual_batch_size,
            adjustment=adjustment,
            name=name,
            **kwargs)

    def call(self, inputs, training=False, use_moving_statistics=True):
        return super(MovingFreeBatchNormalization, self).call(inputs,
                                                              training=training,
                                                              use_moving_statistics=use_moving_statistics)


def moving_free_batch_normalization(inputs,
                                    axis=-1,
                                    momentum=0.99,
                                    epsilon=1e-3,
                                    center=True,
                                    scale=True,
                                    beta_initializer=init_ops.zeros_initializer(),
                                    gamma_initializer=init_ops.ones_initializer(),
                                    mean_initializer=init_ops.zeros_initializer(),
                                    variance_initializer=init_ops.ones_initializer(),
                                    beta_regularizer=None,
                                    gamma_regularizer=None,
                                    beta_constraint=None,
                                    gamma_constraint=None,
                                    training=False,
                                    use_moving_statistics=True,
                                    trainable=True,
                                    name=None,
                                    reuse=None,
                                    fused=None,
                                    virtual_batch_size=None,
                                    adjustment=None):

    layer = MovingFreeBatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        mean_initializer=mean_initializer,
        variance_initializer=variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        fused=fused,
        trainable=trainable,
        virtual_batch_size=virtual_batch_size,
        adjustment=adjustment,
        name=name,
        _reuse=reuse,
        _scope=name)
    return layer.apply(inputs, training=training, use_moving_statistics=use_moving_statistics)


# Aliases

MovingFreeBatchNorm = MovingFreeBatchNormalization
moving_free_batch_norm = moving_free_batch_normalization
