from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils as tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.ops import gen_control_flow_ops


class MovingFreeBatchNormalization(BatchNormalization):

    def build(self, input_shape):

        super(BatchNormalization, self).build(input_shape)
        self.built = False
        # all assertion are

        input_shape = tensor_shape.TensorShape(input_shape)
        ndims = len(input_shape)

        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            param_dtype = dtypes.float32
        else:
            param_dtype = self.dtype or dtypes.float32

        axis_to_dim = {x: input_shape[x].value for x in self.axis}

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
                    self.axis[idx] = x + 1  # Account for added dimension

        try:
            # Disable variable partitioning when creating the moving mean and variance
            if hasattr(self, '_scope') and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None

            # internal statistics fitted during a pre-inference step
            self.mean = self.add_variable(
                name='mean',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.moving_mean_initializer,
                trainable=False)

            self.variance = self.add_variable(
                name='variance',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.moving_variance_initializer,
                trainable=False)

            self.n_updates = self.add_variable(
                name='n_updates',
                shape=[],
                dtype=param_dtype,
                initializer=init_ops.zeros_initializer(),
                trainable=False)

        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def _assign_moving_average(self, variable, value, momentum):
        with ops.name_scope(None, 'AssignMovingAvg',
                            [variable, value, momentum]) as scope:
            decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
            if decay.dtype != variable.dtype.base_dtype:
                decay = math_ops.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - value) * decay
            return state_ops.assign_sub(variable, update_delta, name=scope)

    def _update_statistics(self, variable, value, n_updates):
        with ops.name_scope(None, 'UpdateStatistics',
                            [variable, value, n_updates]) as scope:
            with ops.colocate_with(variable):
                stat = variable * n_updates + value
                stat /= n_updates + 1
                return state_ops.assign(variable, stat, name=scope)

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

        # use_moving_statistics==True use moving_mean and moving_variance, else mean and variance
        mean = tf_utils.smart_cond(use_moving_statistics, lambda: self.moving_mean, lambda: self.mean)
        variance = tf_utils.smart_cond(use_moving_statistics, lambda: self.moving_variance, lambda: self.variance)

        # these variables will be used in _fused_batch_norm_inference(), thanks to python closure

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
        # if training == True: mean and variance returned are mean and variance of the current batch
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

            # add this combination of operations to a specific collection 'UPDATE_BN_OPS'
            ops.add_to_collection('UPDATE_BN_OPS', update_n_updates)

            # operations to reset bn statistics
            reset_mean = state_ops.assign(self.mean, array_ops.zeros_like(self.mean))
            reset_variance = state_ops.assign(self.variance, array_ops.zeros_like(self.variance))
            reset_n_updates = state_ops.assign(self.n_updates, 0.)
            with ops.control_dependencies([reset_mean, reset_variance, reset_n_updates]):
                reset_bn = gen_control_flow_ops.no_op("ResetBatchNormStats")
            ops.add_to_collection('RESET_BN_OPS', reset_bn)

            # to keep the classical behavior of the Batch Norm !
            # update moving averages and add operations to tf.GraphKeys.UPDATE_OPS
            # these operation must be run when optimizing the network
            moving_mean_update = self._assign_moving_average(self.moving_mean, mean, momentum)
            moving_variance_update = self._assign_moving_average(self.moving_variance, variance, momentum)
            self.add_update(moving_mean_update, inputs=True)
            self.add_update(moving_variance_update, inputs=True)

        return output

    def call(self, inputs, training=None, use_moving_statistics=True):
        """

        :param inputs: input features
        :param training: boolean or boolean Tensor (with shape []) which determines the current training phase
        :param use_moving_statistics: boolean or boolean Tensor (with shape []) which selects statistics to use
               when training==True (or the Tensor value) statistics (mean and variance) are from the inputs !
               when training==False, if use_moving_statistics==True -> feed forward with moving statistics (updated
                                        with operations defined in GraphKeys.UPDATE_OPS)
                                     else (use_moving_statistics==False -> feed forward with raw statistics (updated
                                        with operations from collections 'UPDATE_BN_OPS'
                                        'RESET_BN_OPS' contains operations to reset these vaiables between inferences.
        """
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
            return outputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.get_shape()
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

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
                                                                   lambda: self.moving_mean,
                                                                   lambda: self.mean))
            variance = tf_utils.smart_cond(training,
                                           lambda: variance,
                                           lambda: tf_utils.smart_cond(use_moving_statistics,
                                                                       lambda: self.moving_variance,
                                                                       lambda: self.variance))

            if self.renorm:
                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    mean, variance, training)
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
                d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
                scale, offset = _compose_transforms(r, d, scale, offset)
            else:
                new_mean, new_variance = mean, variance

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and moving_variance
                # with each sub-batch. However, since the moving statistics are only
                # used during evaluation, it is more efficient to just update in one
                # step and should not make a significant difference in the result.
                new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)

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
            with ops.control_dependencies([reset_mean, reset_variance, reset_n_updates]):
                reset_bn = gen_control_flow_ops.no_op("ResetBatchNormStats")
            ops.add_to_collection('RESET_OPS', reset_bn)

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

        return outputs


def moving_free_batch_normalization(inputs,
                                    axis=-1,
                                    momentum=0.99,
                                    epsilon=1e-3,
                                    center=True,
                                    scale=True,
                                    beta_initializer=init_ops.zeros_initializer(),
                                    gamma_initializer=init_ops.ones_initializer(),
                                    moving_mean_initializer=init_ops.zeros_initializer(),
                                    moving_variance_initializer=init_ops.ones_initializer(),
                                    beta_regularizer=None,
                                    gamma_regularizer=None,
                                    beta_constraint=None,
                                    gamma_constraint=None,
                                    training=False,
                                    trainable=True,
                                    use_moving_statistics=True,
                                    name=None,
                                    reuse=None,
                                    renorm=False,
                                    renorm_clipping=None,
                                    renorm_momentum=0.99,
                                    fused=None,
                                    virtual_batch_size=None,
                                    adjustment=None):
    """

    :param inputs: input tensor
    :param axis: An `int`, the axis that should be normalized (typically the features
      axis). For instance, after a `Convolution2D` layer with
      `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
    :param momentum: Momentum for the moving average.
    :param epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    :param center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    :param scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    :param beta_initializer: Initializer for the beta weight.
    :param gamma_initializer: Initializer for the gamma weight.
    :param moving_mean_initializer: Initializer for the moving mean and the raw mean (when not using the moving
      statistics).
    :param moving_variance_initializer: Initializer for the moving variance and the raw variance (when not using the
      moving statistics).
    :param beta_regularizer: Optional regularizer for the beta weight.
    :param gamma_regularizer: Optional regularizer for the gamma weight.
    :param beta_constraint: An optional projection function to be applied to the `beta`
        weight after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    :param gamma_constraint: An optional projection function to be applied to the
        `gamma` weight after being updated by an `Optimizer`.
    :param training: Either a Python boolean, or a TensorFlow boolean scalar tensor
      (e.g. a placeholder). Whether to return the output in training mode
      (normalized with statistics of the current batch) or in inference mode
      (normalized with moving statistics). **NOTE**: make sure to set this
      parameter correctly, or else your training/inference will not work
      properly.
    :param trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    :param use_moving_statistics: Either a Python boolean, or a TensorFlow boolean scalar tensor (e.g. a placeholder).
        Whether to use moving statitics or computed statitics in inference mode (training==False).
    :param name: String, the name of the layer.
    :param reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
    :param renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    :param renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    :param renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    :param fused: if `None` or `True`, use a faster, fused implementation if possible.
      If `False`, use the system recommended implementation.
    :param virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
      which means batch normalization is performed across the whole batch. When
      `virtual_batch_size` is not `None`, instead perform "Ghost Batch
      Normalization", which creates virtual sub-batches which are each
      normalized separately (with shared gamma, beta, and moving statistics).
      Must divide the actual batch size during execution.
    :param adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example, if axis==-1,
        `adjustment = lambda shape: (
          tf.random_uniform(shape[-1:], 0.93, 1.07),
          tf.random_uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.
    :return: Output tensor, corresponding to the normalized neural activation
    """

    layer = MovingFreeBatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint,
        renorm=renorm,
        renorm_clipping=renorm_clipping,
        renorm_momentum=renorm_momentum,
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
