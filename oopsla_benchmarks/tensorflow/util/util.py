import time
import os
import numpy as np
import tensorflow as tf

from models import mobilenet
from models import resnet
from models import vgg
from models import dqn
from models import dcgan

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


def log_value(device, backend, task_type, workload, method, template, value, out_file='tmp.log'):
    """
    append experiment result to a central log file

    Parameters
    ----------
    device: str
    backend: str
    task_type: str
    workload: str
    method: str
    template: str
    value: str
    out_file: str
    """
    log_line = "\t".join([str(x) for x in [device, backend, task_type, workload, method, template, value, time.time()]])
    with open(out_file, 'a') as fout:
        fout.write(log_line + "\n")


def log_value_old(target, device, task_name, method, value, outfile='tmp.log'):
    """
    append experiment result to a central log file
    Parameters
    ----------
    target: str
        one of 'cuda', 'opencl', 'llvm'
    device: str
        return string by TVMContext.device_name
    task_name: str
    method: str
    outfile: str
    """

    with open(outfile, 'a') as fout:
        fout.write("\t".join([str(x) for x in
            (target, device, task_name, method, value, time.time())]) + '\n')

def array2str_round(x, decimal=6):
    """ print an array of float number to pretty string with round

    Parameters
    ----------
    x: Array of float or float
    decimal: int
    """
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return "[" + ", ".join([array2str_round(y, decimal=decimal)
                                for y in x]) + "]"
    format_str = "%%.%df" % decimal
    return format_str % x


def query_log_key(target, device, task_name, method, filename='all.log'):
    """ query value from uniform experiment log file
    the records in file should be logged by autotvm-exp/util/util.py log_value

    Parameters
    ----------
    target: str
        one of 'cuda', 'opencl', 'llvm'
    device: str
        return string by TVMContext.device_name
    task_name: str
    method: str
    filename: str
    """
    finds = []
    wanted = ''.join((target, device, task_name, method))
    with open(filename) as fin:
        for line in fin.readlines():
            items = line.split('\t')
            if len(items) != 6:
                continue
            target, device, task_name, method, value, tstamp = items
            key = ''.join((target, device, task_name, method))

            if key == wanted:
                finds.append(value)

    if finds:
        return finds[-1]
    else:
        return None


def query_flop(task_name):
    """
    Query number of float operation of a task.
    use this function to avoid the dependency of autotvm

    Parameters
    ----------
    task_name: string

    Returns
    ------
    flop: int
    """
    res_table = {
        "resnet.C1.B1": 236027904,
        "resnet.C2.B1": 231211008,
        "resnet.C3.B1": 25690112,
        "resnet.C4.B1": 115605504,
        "resnet.C5.B1": 12845056,
        "resnet.C6.B1": 231211008,
        "resnet.C7.B1": 115605504,
        "resnet.C8.B1": 12845056,
        "resnet.C9.B1": 231211008,
        "resnet.C10.B1": 115605504,
        "resnet.C11.B1": 12845056,
        "resnet.C12.B1": 231211008,

        'mobilenet.D1.B1': 7225344,
        'mobilenet.D2.B1': 3612672,
        'mobilenet.D3.B1': 7225344,
        'mobilenet.D4.B1': 1806336,
        'mobilenet.D5.B1': 3612672,
        'mobilenet.D6.B1': 903168,
        'mobilenet.D7.B1': 1806336,
        'mobilenet.D8.B1': 451584,
        'mobilenet.D9.B1': 903168,

        "other.DEN1": 1024 * 1024 * 1024 * 2,
    }

    if task_name.count('.') == 3:
        task_name = task_name[:task_name.rindex('.')]

    return res_table[task_name]

def query_color(color):
    trans_table = {
        'blue':   '#7cb5ec',
        'black':  '#434348',
        'green':  '#90ed7d',
        'orange': '#f7a35c',
        'purple': '#8085e9',
        'brown':  '#8d6e63',
        'pink':   '#f15c80',
    }

    return trans_table[color]

def enhance_color(color, h=1, l=1, s=1):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))

    h, l, s = h * c[0], l * c[1], s * c[2]
    h, l, s = [max(min(x, 1), 0) for x in [h, l, s]]

    return colorsys.hls_to_rgb(h, l, s)

