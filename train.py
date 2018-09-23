import tensorflow as tf
import numpy as np
import os
from math import ceil
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.datasets.cifar10 import load_data as load_cifar10
from tqdm import tqdm
import argparse
import sys
import time
from pprint import  pprint

from stochastic_weight_averaging import StochasticWeightAveraging
from resnet_model import Model

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


def get_best_model(model_dir, mode=None):
    model_to_restore = None
    if mode is None:
        list_best_model_index = glob(os.path.join(model_dir, 'best_model.ckpt-*.index'))
    elif mode == "swa":
        list_best_model_index = glob(os.path.join(model_dir, 'best_model_swa.ckpt-*.index'))
    else:
        raise ValueError("Invalid mode on 'get_best_model' : {}".format(mode))

    if len(list_best_model_index) > 0:
        model_to_restore = list_best_model_index[0].split('.index')[0]
    return model_to_restore


def build_model(inputs, is_training_bn=True, getter=None):

    num_blocks = (RESNET_SIZE - 2) // 6

    with tf.variable_scope('MODEL', custom_getter=getter, reuse=tf.AUTO_REUSE):
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

        logits = model(inputs, training=is_training_bn)
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

    print("... creating a TensorFlow session ...\n")
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("... loading CIFAR10 dataset ...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=0.2,
                                                      shuffle=True,
                                                      stratify=y_train)
    # minimal preprocessing
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)

    print("\tTRAIN - images {} | {}  - labels {} - {}".format(x_train.shape, x_train.dtype, y_train.shape, y_train.dtype))
    print("\tVAL - images {} | {}  - labels {} - {}".format(x_val.shape, x_val.dtype, y_val.shape, y_val.dtype))
    print("\tTEST - images {} | {}  - labels {} - {}\n".format(x_test.shape, x_test.dtype, y_test.shape, y_test.dtype))

    print('... creating TensorFlow datasets ...')
    batch_x, batch_y, handle, handle_train, handle_val, handle_test \
        = build_dataset(sess, x_train, x_val, x_test, y_train, y_val, y_test,
                        batch_size=params.batch_size,
                        buffer_size=x_train.shape[0])

    nb_batches_per_epoch_train = int(ceil(x_train.shape[0]/params.batch_size))
    nb_batches_per_epoch_val = int(ceil(x_val.shape[0]/params.batch_size))
    nb_batches_per_epoch_test = int(ceil(x_test.shape[0]/params.batch_size))

    print('nb_batches_per_epoch_train : {}'.format(nb_batches_per_epoch_train))
    print('nb_batches_per_epoch_val : {}'.format(nb_batches_per_epoch_val))
    print('nb_batches_per_epoch_test : {}\n'.format(nb_batches_per_epoch_test))

    print('... building model ...')
    with tf.name_scope('INPUTS'):
        learning_rate = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')
        is_training_bn = tf.placeholder(shape=[], dtype=tf.bool, name='is_training_bn')
        global_step = tf.train.get_or_create_global_step()
        use_swa_model = tf.placeholder(shape=[], dtype=tf.bool, name='use_swa_model')

    logits = build_model(batch_x, is_training_bn=is_training_bn, getter=None)

    with tf.name_scope('SWA'):
        swa = StochasticWeightAveraging()
        swa_op = swa.apply(var_list=tf.trainable_variables())

        def swa_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            swa_var = swa.average(var)
            return swa_var if swa_var else var

        logits_swa = build_model(batch_x, is_training_bn=is_training_bn, getter=swa_getter)

    # logits = tf.cond(use_swa_model, lambda: logits_swa, lambda: logits, name='cond_swa')

    with tf.name_scope('LOSS'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_y))
        loss_swa = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_swa, labels=batch_y))
        acc = tf.reduce_mean(tf.cast(tf.equal(batch_y, tf.argmax(logits, axis=1)), dtype=tf.float32))
        acc_swa = tf.reduce_mean(tf.cast(tf.equal(batch_y, tf.argmax(logits_swa, axis=1)), dtype=tf.float32))

        loss = tf.cond(use_swa_model, lambda: loss_swa, lambda: loss)
        acc = tf.cond(use_swa_model, lambda: acc_swa, lambda: acc)

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

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            grads_and_vars = opt.compute_gradients(loss, var_list=tf.trainable_variables())

            if 'W' in params.opt:
                # when using AdamW or MomentumW

                if params.weight_decay_on == "all":
                    decay_var_list = tf.trainable_variables()
                elif params.weight_decay_on == "kernels":
                    decay_var_list =[]
                    for var in tf.trainable_variables():
                        if 'kernel' in var.name:
                            decay_var_list.append(var)
                else:
                    raise ValueError('Invalid --weight_decay_on : {}'.format(params.weight_decay_on))

                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step,
                                               decay_var_list=decay_var_list, name='train_op')
            else:
                # without weight decay
                train_op = opt.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')

    with tf.name_scope('METRICS'):

        summaries_vars = []
        for grad, var in grads_and_vars:
            var_name = var.name.split(':')[0]
            summaries_vars.append(tf.summary.histogram('VAR/{}'.format(var_name), var))
            summaries_vars.append(tf.summary.histogram('GRAD/{}'.format(var_name), grad))

        vars_swa = {}
        for var in tf.global_variables():
            shadow_var = swa.average(var)
            if shadow_var is not None:
                vars_swa[shadow_var.name] = shadow_var
        for var in vars_swa.values():
            var_name = var.name.split(':')[0]
            summaries_vars.append(tf.summary.histogram('SHADOW/{}'.format(var_name), var))

        n_models_summary = tf.summary.scalar('n_models', swa.n_models)
        lr_summary = tf.summary.scalar('lr', learning_rate)

        acc_mean, acc_update_op = tf.metrics.mean(acc)
        loss_mean, loss_update_op = tf.metrics.mean(loss)

        acc_summary = tf.summary.scalar('acc_train', acc)
        loss_summary = tf.summary.scalar('loss_train', loss)

        acc_mean_summary = tf.summary.scalar('acc_mean', acc_mean)
        loss_mean_summary = tf.summary.scalar('loss_mean', loss_mean)

        summaries_mean = tf.summary.merge([acc_mean_summary, loss_mean_summary], name='summaries_mean')
        summaries = tf.summary.merge([acc_summary, loss_summary, n_models_summary, lr_summary]+summaries_vars,
                                     name='summaries')



    with tf.name_scope('INIT_OPS'):
        global_init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        sess.run(global_init_op)
        sess.run(local_init_op)

    vars_classic = [var for var in tf.global_variables() if 'StochasticWeightAveraging' not in var.name]
    best_saver = tf.train.Saver(vars_classic, max_to_keep=1)

    vars_swa = {}
    for var in tf.global_variables():
        shadow_var = swa.average(var)
        if shadow_var is not None:
            vars_swa[shadow_var.name] = shadow_var
        else:
            vars_swa[var.name] = var

    best_saver_swa = tf.train.Saver(list(vars_swa.values()), max_to_keep=1)

    with tf.name_scope('FILE_WRITERS'):
        writer_train = tf.summary.FileWriter(os.path.join(log_dir, 'train'), graph=sess.graph)
        writer_val = tf.summary.FileWriter(os.path.join(log_dir, 'val'))
        writer_test = tf.summary.FileWriter(os.path.join(log_dir, 'test'))
        if params.use_swa:
            writer_val_swa = tf.summary.FileWriter(os.path.join(log_dir, 'val_swa'))
            writer_test_swa = tf.summary.FileWriter(os.path.join(log_dir, 'test_swa'))

    if params.strategy_lr == "constant":

        def get_learning_rate(step, epoch, steps_per_epoch):
            return params.init_lr
    elif params.strategy_lr == "cyclical":

        def get_learning_rate(step, epoch, steps_per_epoch):
            if epoch < params.epochs_before_swa:
                return params.init_lr

            total_steps_per_cycle = params.cycle_length * steps_per_epoch
            step = (step - params.epochs_before_swa * steps_per_epoch) % total_steps_per_cycle
            lr = (params.alpha1_lr - params.alpha2_lr) * (1 - step / total_steps_per_cycle) + params.alpha2_lr
            return lr
    elif params.strategy_lr == "cosine":

        def get_learning_rate(step, epoch, steps_per_epoch):
            if epoch < params.epochs_before_swa:
                return params.init_lr
            total_steps_per_cycle = params.cycle_length * steps_per_epoch
            step = (step - params.epochs_before_swa * steps_per_epoch) % total_steps_per_cycle
            lr = (params.alpha1_lr - params.alpha2_lr) * np.cos(np.pi*(step/total_steps_per_cycle)) + params.alpha2_lr
            return lr
    else:
        raise ValueError('Invalid --strategy_lr : {}'.format(params.strategy_lr))

    feed_dict_train = {is_training_bn: True, handle: handle_train, use_swa_model: False}

    best_acc = 0.
    best_step = 0
    best_epoch = 0

    best_acc_swa = 0.
    best_step_swa = 0
    best_epoch_swa = 0

    step = -1

    for epoch in range(params.epochs):

        # validation step before SWA op
        # re-initialize local variables
        sess.run(local_init_op)

        feed_dict = {is_training_bn: False, handle: handle_val, use_swa_model: False}

        for _ in tqdm(range(nb_batches_per_epoch_val), desc='VALIDATION @ EPOCH {}'.format(epoch)):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_val.add_summary(s, global_step=step)
        writer_val.flush()
        print("VALIDATION @ EPOCH {} | without SWA : acc={:.4f}  loss={:.5f}".format(epoch, acc_v, loss_v))

        if acc_v > best_acc:
            print("\tNew best model !")
            best_acc = acc_v
            best_step = step
            best_epoch = epoch
            ckpt_path = os.path.join(log_dir, 'best_model.ckpt')
            best_saver.save(sess, ckpt_path, global_step=step)

        if epoch >= params.epochs_before_swa and params.use_swa and (epoch-params.epochs_before_swa) % params.cycle_length == 0:

            # apply SWA
            sess.run(swa_op)
            feed_dict = {is_training_bn: False, handle: handle_val, use_swa_model: True}

            # re-initialize local variables
            sess.run(local_init_op)
            for _ in tqdm(range(nb_batches_per_epoch_val), desc='VALIDATION with SWA @ EPOCH {}'.format(epoch)):
                sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

            acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
            writer_val_swa.add_summary(s, global_step=step)
            writer_val_swa.flush()
            print("VALIDATION @ EPOCH {} | with SWA : acc={:.4f}  loss={:.5f}".format(epoch, acc_v, loss_v))

            if acc_v > best_acc_swa:
                print("\tNew best model !")
                best_acc_swa = acc_v
                best_step_swa = step
                best_epoch_swa = epoch
                ckpt_path = os.path.join(log_dir, 'best_model_swa.ckpt')
                best_saver_swa.save(sess, ckpt_path, global_step=step)

        # re-initialize local variables
        sess.run(local_init_op)

        # perform 1 epoch on training data
        for _ in tqdm(range(nb_batches_per_epoch_train), desc='TRAIN @ EPOCH {}'.format(epoch)):
            step += 1

            feed_dict_train[learning_rate] = get_learning_rate(step, epoch, nb_batches_per_epoch_train)

            if step % PLOT_PERIOD == 0:
                _, s, _, _ = sess.run([train_op, summaries, acc_update_op, loss_update_op], feed_dict=feed_dict_train)
                writer_train.add_summary(s, global_step=step)

            else:
                sess.run([train_op, acc_update_op, loss_update_op], feed_dict=feed_dict_train)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean], feed_dict=feed_dict)
        writer_train.add_summary(s, global_step=step)
        writer_train.flush()
        print("TRAIN @ EPOCH {} | : acc={:.4f}  loss={:.5f}".format(epoch, acc_v, loss_v))

    if best_acc > 0.:

        print("Load best model without SWA  |  ACC={:.5f} form epoch={}".format(best_acc, best_epoch))
        model_to_restore = get_best_model(log_dir)
        if model_to_restore is not None:
            best_saver.restore(sess, model_to_restore)
        else:
            print("Impossible to load best model .... ")

        sess.run(local_init_op)
        feed_dict = {is_training_bn: False, handle: handle_test, use_swa_model: False}
        for _ in tqdm(range(nb_batches_per_epoch_test), desc='TEST'):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_test.add_summary(s, global_step=best_step)
        writer_test.flush()
        print("TEST @ EPOCH {} | without SWA : acc={:.4f}  loss={:.5f}".format(best_epoch, acc_v, loss_v))

    if best_acc_swa > 0. and params.use_swa:

        print("Load best model with SWA  |  ACC={:.5f} form epoch={}".format(best_acc_swa, best_epoch_swa))
        model_to_restore = get_best_model(log_dir, mode='swa')
        if model_to_restore is not None:
            best_saver_swa.restore(sess, model_to_restore)
        else:
            print("Impossible to load best model .... ")

        sess.run(local_init_op)
        feed_dict = {is_training_bn: False, handle: handle_test, use_swa_model: True}
        for _ in tqdm(range(nb_batches_per_epoch_test), desc='TEST'):
            sess.run([acc_update_op, loss_update_op], feed_dict=feed_dict)

        acc_v, loss_v, s = sess.run([acc_mean, loss_mean, summaries_mean])
        writer_test_swa.add_summary(s, global_step=best_step_swa)
        writer_test_swa.flush()
        print("TEST @ EPOCH {} | with SWA : acc={:.4f}  loss={:.5f}".format(best_epoch_swa, acc_v, loss_v))

    writer_train.close()
    writer_val.close()
    writer_test.close()
    if params.use_swa:
        writer_val_swa.close()
        writer_test_swa.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', dest='log_dir', type=str, default='test')

    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128)

    parser.add_argument('--opt', dest='opt', type=str, default='momentumW') # adam, adamW, momentum, momentumW
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-4)
    parser.add_argument('--weight_decay_on', dest='weight_decay_on', type=str, default='all') # all or kernels

    parser.add_argument('--use_swa', dest='use_swa', type=int, default=1)
    parser.add_argument('--epochs_before_swa', dest='epochs_before_swa', type=int, default=10)
    parser.add_argument('--strategy_lr', dest='strategy_lr', type=str, default='cyclical') # constant, cyclical, cosine
    parser.add_argument('--cycle_length', dest='cycle_length', type=int, default=4) # in epochs

    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.005)
    parser.add_argument('--alpha1_lr', dest='alpha1_lr', type=float, default=0.05)
    parser.add_argument('--alpha2_lr', dest='alpha2_lr', type=float, default=0.0005)

    params = parser.parse_args(sys.argv[1:])
    main(params)