import tensorflow as tf
import numpy as np
import os
from math import ceil
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras.datasets.cifar10 import load_data as load_cifar10
from tqdm import tqdm
import argparse
import sys

from swa_tf import StochasticWeightAveraging
from resnet_model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MAIN_LOG_DIR = 'logs/'
INTERPOLATION = "BILINEAR"

PLOT_PERIOD = 20

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3


DATA_FORMAT = "channels_first"
RESNET_SIZE = 32
BOTTLENECK = False
RESNET_VERSION = 2

COLORS = {
    'green': ['\033[32m', '\033[39m'],
    'red': ['\033[31m', '\033[39m']
}


def get_best_model(model_dir, model='best_model'):
    model_to_restore = None
    list_best_model_index = glob(os.path.join(model_dir, '{}.ckpt-*.index'.format(model)))
    if len(list_best_model_index) > 0:
        model_to_restore = list_best_model_index[0].split('.index')[0]
    return model_to_restore


def build_model(inputs, is_training_bn=True, use_moving_statistics=True):

    num_blocks = (RESNET_SIZE - 2) // 6

    model = Model(resnet_size=RESNET_SIZE,
                  bottleneck=BOTTLENECK,
                  num_classes=10,
                  num_filters=16,
                  kernel_size=3,
                  conv_stride=1,
                  first_pool_size=None,
                  first_pool_stride=None,
                  block_sizes=[num_blocks, ]*3,
                  block_strides=[1, 2, 3],
                  final_size=64,
                  resnet_version=RESNET_VERSION,
                  data_format=DATA_FORMAT,
                  dtype=tf.float32)
    logits = model(inputs, training=is_training_bn, use_moving_statistics=use_moving_statistics)
    return logits


def map_func_train(image, label):

    image = tf.image.resize_image_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)

    return image, label


def map_func_val_test(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label


def build_dataset(sess, x_train_np, x_val_np, x_test_np, y_train_np, y_val_np, y_test_np,
                  batch_size=32,
                  buffer_size=40000):

    with tf.variable_scope('DATA'):

        with tf.name_scope('dataset_placeholders'):
            x_train_tf = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='X_train')
            x_val_tf = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='X_val')
            x_test_tf = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='X_test')

            y_train_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_train')
            y_val_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_val')
            y_test_tf = tf.placeholder(shape=[None, ], dtype=tf.int64, name='y_test')

        with tf.name_scope('dataset_train'):
            dataset_train = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_tf))
            dataset_train = dataset_train.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
            dataset_train = dataset_train.repeat()
            dataset_train = dataset_train.map(map_func_train, num_parallel_calls=os.cpu_count()//2)
            dataset_train = dataset_train.batch(batch_size=batch_size)

        with tf.name_scope('dataset_val'):
            dataset_val = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val_tf))
            dataset_val = dataset_val.map(map_func_val_test, num_parallel_calls=os.cpu_count()//2)
            dataset_val = dataset_val.batch(batch_size=batch_size)
            dataset_val = dataset_val.repeat()

        with tf.name_scope('dataset_test'):
            dataset_test = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test_tf))
            dataset_test = dataset_test.map(map_func_val_test, num_parallel_calls=os.cpu_count()//2)
            dataset_test = dataset_test.batch(batch_size=batch_size)
            dataset_test = dataset_test.repeat()

        with tf.name_scope('iterators'):
            handle = tf.placeholder(name='handle', shape=[], dtype=tf.string)
            iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
            batch_x, batch_y = iterator.get_next()

            iterator_train = dataset_train.make_initializable_iterator()
            iterator_val = dataset_val.make_initializable_iterator()
            iterator_test = dataset_test.make_initializable_iterator()

        handle_train = sess.run(iterator_train.string_handle())
        handle_val = sess.run(iterator_val.string_handle())
        handle_test = sess.run(iterator_test.string_handle())

        print('...initialize datasets...')
        sess.run(iterator_train.initializer, feed_dict={x_train_tf: x_train_np, y_train_tf: y_train_np})
        sess.run(iterator_val.initializer, feed_dict={x_val_tf: x_val_np, y_val_tf: y_val_np})
        sess.run(iterator_test.initializer, feed_dict={x_test_tf: x_test_np, y_test_tf: y_test_np})

    return batch_x, batch_y, handle, handle_train, handle_val, handle_test


def main(params):

    log_dir = os.path.join(MAIN_LOG_DIR, params.log_dir)
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    tf.set_random_seed(seed=42)

    print("... creating a TensorFlow session ...\n")
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("... loading CIFAR10 dataset ...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    x_train, y_train = shuffle(x_train, y_train)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2,
                                                      stratify=y_train,
                                                      random_state=51)
    # cast samples and labels
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    print("\tTRAIN - images {} | {}  - labels {} - {}".format(x_train.shape, x_train.dtype, y_train.shape, y_train.dtype))
    print("\tVAL - images {} | {}  - labels {} - {}".format(x_val.shape, x_val.dtype, y_val.shape, y_val.dtype))
    print("\tTEST - images {} | {}  - labels {} - {}\n".format(x_test.shape, x_test.dtype, y_test.shape, y_test.dtype))

    print('... creating TensorFlow datasets ...\n')
    batch_x, batch_y, handle, handle_train, handle_val, handle_test \
        = build_dataset(sess, x_train, x_val, x_test, y_train, y_val, y_test,
                        batch_size=params.batch_size,
                        buffer_size=x_train.shape[0])

    nb_batches_per_epoch_train = int(ceil(x_train.shape[0]/params.batch_size))
    nb_batches_per_epoch_val = int(ceil(x_val.shape[0]/params.batch_size))
    nb_batches_per_epoch_test = int(ceil(x_test.shape[0]/params.batch_size))
    print('\tnb_batches_per_epoch_train : {}'.format(nb_batches_per_epoch_train))
    print('\tnb_batches_per_epoch_val : {}'.format(nb_batches_per_epoch_val))
    print('\tnb_batches_per_epoch_test : {}\n'.format(nb_batches_per_epoch_test))

    print('... building model ...\n')
    with tf.name_scope('INPUTS'):
        learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        is_training_bn = tf.placeholder(shape=[], dtype=tf.bool, name='is_training_bn')
        use_moving_statistics = tf.placeholder(shape=[], dtype=tf.bool, name='use_moving_statistics')
        global_step = tf.train.get_or_create_global_step()

    logits = build_model(batch_x, is_training_bn=is_training_bn, use_moving_statistics=use_moving_statistics)
    model_vars = tf.trainable_variables()

    update_bn_ops = tf.group(*tf.get_collection('UPDATE_BN_OPS'))
    reset_bn_ops = tf.group(*tf.get_collection('RESET_BN_OPS'))

    with tf.name_scope('LOSS'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
        acc = tf.reduce_mean(tf.cast(tf.equal(batch_y, tf.argmax(logits, axis=1)), dtype=tf.float32))

    with tf.name_scope('OPTIMIZER'):
        if params.opt == "adam":
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif params.opt == "momentum":
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=params.momentum)
        elif params.opt == "adamW":
            opt = tf.contrib.opt.AdamWOptimizer(weight_decay=params.weight_decay, learning_rate=learning_rate,
                                                beta1=0.9, beta2=0.999, epsilon=1e-8)
        elif params.opt == "momentumW":
            opt = tf.contrib.opt.MomentumWOptimizer(weight_decay=params.weight_decay, learning_rate=learning_rate,
                                                    momentum=params.momentum)
        else:
            raise ValueError('Invalid --opt argument : {}'.format(params.opt))

        if 'W' in params.opt:
            # when using AdamW or MomentumW

            if params.weight_decay_on == "all":
                decay_var_list = tf.trainable_variables()
            elif params.weight_decay_on == "kernels":
                decay_var_list = []
                for var in tf.trainable_variables():
                    if 'kernel' in var.name:
                        decay_var_list.append(var)
            else:
                raise ValueError('Invalid --weight_decay_on : {}'.format(params.weight_decay_on))

        # force updates of moving averages in BN before optimizing the network
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            grads_and_vars = opt.compute_gradients(loss, var_list=tf.trainable_variables())
            # in case of moving free batch normalization, there is no control dependencies on moving means/variances
            if 'W' in params.opt:
                # add decay_var_list argument for decoupled optimizers
                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step,
                                               decay_var_list=decay_var_list, name='train_op')
            else:
                # without weight decay
                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')

    if params.use_swa:
        with tf.name_scope('SWA'):
            swa = StochasticWeightAveraging()
            swa_op = swa.apply(var_list=model_vars)
            # Make backup variables
            with tf.variable_scope('BackupVariables'):
                backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False,
                                               initializer=var.initialized_value())
                               for var in model_vars]
            # operation to assign SWA weights to model
            swa_to_weights = tf.group(*(tf.assign(var, swa.average(var).read_value()) for var in model_vars))
            # operation to store model into backup variables
            save_weight_backups = tf.group(*(tf.assign(bck, var.read_value()) for var, bck in zip(model_vars, backup_vars)))
            # operation to get back values from backup variables to model
            restore_weight_backups = tf.group(*(tf.assign(var, bck.read_value()) for var, bck in zip(model_vars, backup_vars)))

    with tf.name_scope('METRICS'):
        acc_mean, acc_update_op = tf.metrics.mean(acc)
        loss_mean, loss_update_op = tf.metrics.mean(loss)

    # summaries which track loss/acc per batch
    acc_summary = tf.summary.scalar('TRAIN/acc', acc)
    loss_summary = tf.summary.scalar('TRAIN/loss', loss)

    # summaries which track accumulated loss/acc
    acc_mean_summary = tf.summary.scalar('MEAN/acc', acc_mean)
    loss_mean_summary = tf.summary.scalar('MEAN/loss', loss_mean)

    lr_summary = tf.summary.scalar('lr', learning_rate)

    # summaries to plot at each epoch
    summaries_mean = tf.summary.merge([acc_mean_summary, loss_mean_summary], name='summaries_mean')

    # summaries to plot regularly
    summaries = [acc_summary, loss_summary, lr_summary]
    if params.use_swa:
        n_models_summary = tf.summary.scalar('n_models', swa.n_models)
        summaries.append(n_models_summary)
    summaries = tf.summary.merge(summaries, name='summaries')

    with tf.name_scope('INIT_OPS'):
        # local init_ops contains operations to reset to zero accumulators of 'acc' and 'loss'
        global_init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        sess.run(global_init_op)
        sess.run(local_init_op)

    with tf.name_scope('SAVERS'):
        best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        if params.use_swa:
            best_saver_swa = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.name_scope('FILE_WRITERS'):
        writer_train = tf.summary.FileWriter(os.path.join(log_dir, 'train'), graph=sess.graph)
        writer_val = tf.summary.FileWriter(os.path.join(log_dir, 'val'))
        writer_val_bn = tf.summary.FileWriter(os.path.join(log_dir, 'val_bn'))
        writer_test = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

        if params.use_swa:
            writer_val_swa = tf.summary.FileWriter(os.path.join(log_dir, 'val_swa'))
            writer_test_swa = tf.summary.FileWriter(os.path.join(log_dir, 'test_swa'))

    if params.strategy_lr == "constant":
        def get_learning_rate(step, epoch, steps_per_epoch):
            return params.init_lr
    elif params.strategy_lr == "swa":
        def get_learning_rate(step, epoch, steps_per_epoch):
            if epoch < params.epochs_before_swa:
                return params.init_lr

            if not params.use_swa:
                return params.init_lr

            if step > int(0.9 * params.epochs * steps_per_epoch):
                return params.alpha2_lr

            length_slope = int(0.9 * params.epochs * steps_per_epoch) - params.epochs_before_swa * steps_per_epoch
            return params.alpha1_lr - ((params.alpha1_lr - params.alpha2_lr) / length_slope) * \
                                      (step - params.epochs_before_swa * steps_per_epoch)
    else:
        raise ValueError('Invalid --strategy_lr : {}'.format(params.strategy_lr))

    def fit_bn_statistics(epoch, swa=False):
        # re_initialize statistics
        sess.run(reset_bn_ops)

        # when  is_training_bn is True ---> the value fed to use_moving_statistics does not matter !
        feed_dict = {handle: handle_train, is_training_bn: True, use_moving_statistics:True}

        if swa:
            desc = 'FIT STATISTICS @ EPOCH {} for SWA'.format(epoch)
        else:
            desc = 'FIT STATISTICS @ EPOCH {}'.format(epoch)

        for _ in tqdm(range(nb_batches_per_epoch_train), desc=desc):
            sess.run([update_bn_ops], feed_dict=feed_dict)

    def inference(epoch, step, best_acc, best_step, best_epoch, with_moving_statistics=True):
        sess.run(local_init_op)

        feed_dict = {is_training_bn: False, handle: handle_val, use_moving_statistics: with_moving_statistics}

        for _ in tqdm(range(nb_batches_per_epoch_val),
                      desc='VALIDATION (moving_statistics:{}) @ EPOCH {}'.format(with_moving_statistics, epoch)):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])

        if with_moving_statistics:
            writer_val.add_summary(s, global_step=step)
            writer_val.flush()
        else:
            writer_val_bn.add_summary(s, global_step=step)
            writer_val_bn.flush()

        if acc_v > best_acc:
            color = COLORS['green']
            best_acc = acc_v
            best_step = step
            best_epoch = epoch
            ckpt_path = os.path.join(log_dir, 'best_model.ckpt')
            best_saver.save(sess, ckpt_path, global_step=step)
        else:
            color = COLORS['red']

        print("VALIDATION (moving_statistics:{}) @ EPOCH {} | without SWA : {}acc={:.4f}{}  loss={:.5f}".format(with_moving_statistics,
                                                                                                                epoch,
                                                                                                                color[0], acc_v, color[1],
                                                                                                                loss_v))

        return best_acc, best_step, best_epoch

    def inference_swa(epoch, step, best_acc_swa, best_step_swa, best_epoch_swa):
        sess.run(swa_op)
        sess.run(save_weight_backups)
        sess.run(swa_to_weights)

        fit_bn_statistics(epoch, swa=True)

        feed_dict = {is_training_bn: False, handle: handle_val, use_moving_statistics: False}

        # re-initialize local variables
        sess.run(local_init_op)

        # now perform a validation loop
        for _ in tqdm(range(nb_batches_per_epoch_val), desc='VALIDATION with SWA @ EPOCH {}'.format(epoch)):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_val_swa.add_summary(s, global_step=step)
        writer_val_swa.flush()

        if acc_v > best_acc_swa:
            color = COLORS['green']
            best_acc_swa = acc_v
            best_step_swa = step
            best_epoch_swa = epoch
            ckpt_path = os.path.join(log_dir, 'best_model_swa.ckpt')
            best_saver_swa.save(sess, ckpt_path, global_step=step)
        else:
            color = COLORS['red']

        print("VALIDATION @ EPOCH {} | with SWA : {}acc={:.4f}{}  loss={:.5f}".format(epoch,
                                                                                      color[0], acc_v, color[1],
                                                                                      loss_v))

        # now restore regular weights (and SWA ones, but unchanged) from last_saver
        sess.run(restore_weight_backups)

        return best_acc_swa, best_step_swa, best_epoch_swa

    feed_dict_train = {is_training_bn: True, handle: handle_train, use_moving_statistics: True}

    best_acc = 0.
    best_step = 0
    best_epoch = 0

    best_acc_swa = 0.
    best_step_swa = 0
    best_epoch_swa = 0
    step = -1

    # inference with trained variables
    best_acc, best_step, best_epoch = inference(0, 0, best_acc, best_step, best_epoch, with_moving_statistics=True)

    for epoch in range(1, params.epochs+1):
        # ####################################### TRAIN 1 EPOCH ######################################################
        # re-initialize local variables
        sess.run(local_init_op)

        for _ in tqdm(range(nb_batches_per_epoch_train), desc='TRAIN @ EPOCH {}'.format(epoch)):
            step += 1

            feed_dict_train[learning_rate] = get_learning_rate(step, epoch, nb_batches_per_epoch_train)

            if step % PLOT_PERIOD == 0:
                _, s, _, _ = sess.run([train_op, summaries, acc_update_op, loss_update_op], feed_dict=feed_dict_train)
                writer_train.add_summary(s, global_step=step)

            else:
                sess.run([train_op, acc_update_op, loss_update_op], feed_dict=feed_dict_train)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_train.add_summary(s, global_step=step)
        writer_train.flush()
        print("TRAIN @ EPOCH {} | : acc={:.4f}  loss={:.5f}".format(epoch, acc_v, loss_v))
        # ############################################################################################################

        # ###################################### INFERENCE ###########################################################
        # perform inference with trained variables and moving statistics
        best_acc, best_step, best_epoch = inference(epoch, step, best_acc, best_step, best_epoch,
                                                    with_moving_statistics=True)

        # fit mean and var in BN layers, and make inference again
        fit_bn_statistics(epoch, swa=False)
        best_acc, best_step, best_epoch = inference(epoch, step, best_acc, best_step, best_epoch,
                                                    with_moving_statistics=False)

        # perform inference with SWA variables
        if epoch >= params.epochs_before_swa \
                and params.use_swa \
                and (epoch-params.epochs_before_swa) % params.cycle_length == 0:

            # weights are replaced and bn statistics are updated within the function
            best_acc_swa, best_step_swa, best_epoch_swa = inference_swa(epoch, step, best_acc_swa,
                                                                        best_step_swa, best_epoch_swa)
        # ############################################################################################################

    # Inference on test set with trained weights
    if best_acc > 0.:

        print("Load best model without SWA  |  ACC={:.5f} form epoch={}".format(best_acc, best_epoch))
        model_to_restore = get_best_model(log_dir, model='best_model')
        if model_to_restore is not None:
            best_saver.restore(sess, model_to_restore)
        else:
            print("Impossible to load best model .... ")

        sess.run(local_init_op)
        feed_dict = {is_training_bn: False, handle: handle_test, use_moving_statistics: True}
        for _ in tqdm(range(nb_batches_per_epoch_test), desc='TEST'):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_test.add_summary(s, global_step=best_step)
        writer_test.flush()
        print("TEST @ EPOCH {} | without SWA : acc={:.4f}  loss={:.5f}".format(best_epoch, acc_v, loss_v))

    # Inference on test set with trained weights
    if best_acc > 0.:

        print("Load best model without SWA  |  ACC={:.5f} form epoch={}".format(best_acc, best_epoch))
        model_to_restore = get_best_model(log_dir, model='best_model')
        if model_to_restore is not None:
            best_saver.restore(sess, model_to_restore)
        else:
            print("Impossible to load best model .... ")

        fit_bn_statistics(epoch, swa=False)
        sess.run(local_init_op)
        feed_dict = {is_training_bn: False, handle: handle_test, use_moving_statistics: False}
        for _ in tqdm(range(nb_batches_per_epoch_test), desc='TEST'):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_test.add_summary(s, global_step=best_step)
        writer_test.flush()
        print("TEST @ EPOCH {} | without SWA : acc={:.4f}  loss={:.5f}".format(best_epoch, acc_v, loss_v))

    # inference on test set with SWA weights
    if best_acc_swa > 0. and params.use_swa:

        print("Load best model with SWA  |  ACC={:.5f} form epoch={}".format(best_acc_swa, best_epoch_swa))
        model_to_restore = get_best_model(log_dir, model='best_model_swa')
        if model_to_restore is not None:
            # regular weights are already set to SWA weights ... no need to run 'retrieve_swa_weights' op.
            # and BN statistics are already set correctly
            best_saver_swa.restore(sess, model_to_restore)
        else:
            print("Impossible to load best model .... ")

        fit_bn_statistics(epoch, swa=False)
        sess.run(local_init_op)
        feed_dict = {is_training_bn: False, handle: handle_test, use_moving_statistics: False}
        for _ in tqdm(range(nb_batches_per_epoch_test), desc='TEST'):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_test_swa.add_summary(s, global_step=best_step_swa)
        writer_test_swa.flush()
        print("TEST @ EPOCH {} | with SWA : acc={:.4f}  loss={:.5f}".format(best_epoch_swa, acc_v, loss_v))

    writer_train.close()
    writer_val.close()
    writer_val_bn.close()
    writer_test.close()
    if params.use_swa:
        writer_val_swa.close()
        writer_test_swa.close()

    sess.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', dest='log_dir', type=str, default='resnet34-swa')

    parser.add_argument('--epochs', dest='epochs', type=int, default=170)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)

    parser.add_argument('--opt', dest='opt', type=str, default='momentumW') # adam, adamW, momentum, momentumW
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay_on', dest='weight_decay_on', type=str, default='all') # all or kernels

    parser.add_argument('--use_swa', dest='use_swa', type=int, default=1)
    parser.add_argument('--epochs_before_swa', dest='epochs_before_swa', type=int, default=75)
    parser.add_argument('--strategy_lr', dest='strategy_lr', type=str, default='swa') # constant, swa
    parser.add_argument('--cycle_length', dest='cycle_length', type=int, default=1) # in epochs

    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.01)
    parser.add_argument('--alpha1_lr', dest='alpha1_lr', type=float, default=0.01)
    parser.add_argument('--alpha2_lr', dest='alpha2_lr', type=float, default=0.0001)

    params = parser.parse_args(sys.argv[1:])
    main(params)