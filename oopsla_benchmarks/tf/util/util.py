import time
import os
import numpy as np
import tensorflow as tf

from oopsla_benchmarks.tf.models import mobilenet
from oopsla_benchmarks.tf.models import resnet
from oopsla_benchmarks.tf.models import vgg
from oopsla_benchmarks.tf.models import dqn
from oopsla_benchmarks.tf.models import dcgan

TVM_NAME = 'AutoTVM'

# some bug in tensorflow causes it to continue to default to GPU fallback
# implementations even if the device is set to '/cpu:0' so it's necessary
# to trick TF into thinking there is no GPU available to run on CPUs
# (use the object returned inside a with statement)
def no_visible_gpus():
    class GPUHidden():
        def __enter__(self):
            os.putenv('CUDA_VISIBLE_DEVICES', '')

        def __exit__(self, exc_type, exc_value, traceback):
            os.unsetenv('CUDA_VISIBLE_DEVICES')

    return GPUHidden()


# returns a network and image shape based on the network name given
def instantiate_network(network, batch_size, data_format):
    image_shape = (batch_size, 224, 224, 3)

    if network == 'resnet-18':
        net = resnet.imagenet_resnet_v2(resnet_size=18, num_classes=1000, data_format=data_format)
    elif network == 'mobilenet':
        net = mobilenet.wrapped_partial(mobilenet.mobilenet_v1, depth_multiplier=1.0,
                                        scope="%d" % (int(np.random.randint(1 << 31))))
    elif network == 'vgg-16':
        net = mobilenet.wrapped_partial(vgg.vgg_16,
                                        scope="%d" % (int(np.random.randint(1 << 31))))
    elif network == 'nature-dqn':
        net = mobilenet.wrapped_partial(dqn.nature_dqn,
                                        scope="%d" % (int(np.random.randint(1 << 31))))
        image_shape = (batch_size, 84, 84, 4)
    elif network == 'dcgan':
        net = mobilenet.wrapped_partial(dcgan.dcgan, oshape=(32, 32, 3), batch_size=batch_size,
                                        scope="%d" % (int(np.random.randint(1 << 31))))
        image_shape = (batch_size, 100)

    return (net, image_shape)


def cnn_setup(network, dev, batch_size, enable_xla):
    # for CPU, the data format must be channels_last because certain
    # channels_first implementations exist only for GPU
    data_format = 'channels_first' if dev == '/gpu:0' else 'channels_last'
    net, image_shape = instantiate_network(network, batch_size, data_format)

    # if we want to run on CPU we have to trick TF into believing
    # there is no GPU available at all to prevent "fallback" GPU
    # implementations from running (this is a TF bug)
    if dev == '/cpu:0':
        os.putenv('CUDA_VISIBLE_DEVICES', '')

    config = tf.ConfigProto(log_device_placement=False)
    if enable_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.device(dev):
        inputs = tf.constant(np.random.randn(*image_shape).astype(np.float32))
        output = net(inputs, is_training=False)

    sess = tf.Session(config=config)
    sess.__enter__()
    sess.run(tf.global_variables_initializer())
    return [dev, sess, output]


def cnn_trial(dev, sess, output):
    sess.run(output)


def cnn_teardown(dev, sess, output):
    sess.__exit__(None, None, None)
    if dev == '/cpu:0':
        os.unsetenv('CUDA_VISIBLE_DEVICES')


# runs the given cnn in tensorflow on random input and returns the
# score (images/time)
def score_cnn(network, data_format, dev, batch_size, num_batches, enable_xla):
    net, image_shape = instantiate_network(network, batch_size, data_format)

    config = tf.ConfigProto(log_device_placement=False)
    if enable_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.device(dev):
        inputs = tf.constant(np.random.randn(*image_shape).astype(np.float32))
        output = net(inputs, is_training=False)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dry_run = 8
        for i in range(dry_run + num_batches):
            if i == dry_run:
                tic = time.time()
            out = sess.run(output)
        end = time.time()

    return num_batches * batch_size / (end - tic)


# reports the average inference time for a single RNN cell
# cell_wkl is a tuple of task name, cell type (LSTM or RNN), sequence length, hidden size, and voc size
def measure_rnn_cell(cell_wkl, dev, xla, n_times):
    task_name, cell_type, batch_size, num_layer, seq_len, hidden_size, voc_size = cell_wkl

    if cell_type == 'rnn':
        cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    elif cell_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size, hidden_size, state_is_tuple=True)
    elif cell_type == 'basic_lstm':
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
    else:
        raise Exception('Unknown network! '+network_type)

    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layer, state_is_tuple=True)

    with tf.variable_scope(task_name, reuse=tf.AUTO_REUSE):
        with tf.device(dev):
            init_state = cell.zero_state(batch_size, tf.float32)
            if voc_size:
                # encoder
                W = tf.get_variable('W', [voc_size, hidden_size])
                ids = [tf.constant(np.random.randint(0, voc_size, [batch_size]).astype(np.int32)) for _ in range(seq_len)]
                data = [tf.nn.embedding_lookup(W, x) for x in ids]
            else:
                data = [tf.constant(np.random.randn(batch_size, hidden_size).astype(np.float32)) for _ in range(seq_len)]

            output, _cell_state = tf.nn.static_rnn(cell, data, initial_state=init_state)

            if voc_size:
                with tf.variable_scope('softmax'):
                    W = tf.get_variable('W', [hidden_size, voc_size])
                    b = tf.get_variable('b', [voc_size])
                output = [tf.matmul(x, W) + b for x in output]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    config = tf.ConfigProto(log_device_placement=False)
    if xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dry_run = 5
        for i in range(dry_run + n_times):
            if i == dry_run:
                tic = time.time()
            out = sess.run(output)
        end = time.time()

    return (end - tic) / n_times
