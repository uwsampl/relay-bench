import time
import tvm
import mxnet as mx
import numpy as np
from collections import namedtuple

from oopsla_benchmarks.mx.models import mxnet_zoo

def get_network(name):
    image_shape = (1, 3, 224, 224)
    if 'vgg' in name:
        mx_sym = mxnet_zoo.mx_vgg(16)
    elif 'resnet' in name:
        mx_sym = mxnet_zoo.mx_resnet(18)
    elif 'dcgan' in name:
        mx_sym = mxnet_zoo.mx_dcgan()
    elif 'nature-dqn' in name:
        mx_sym = mxnet_zoo.mx_dqn()
    else:
        raise ValueError("Unsupported network: " + name)

    return mx_sym, image_shape


def cnn_setup(network, dev, batch_size):
    mx_sym, image_shape = get_network(network)
    ctx = mx.gpu(0) if dev == 'gpu' else mx.cpu()
    mx_mod = mx.mod.Module(mx_sym, label_names=None, context=ctx)
    mx_mod.bind(data_shapes=[('data', image_shape)], for_training=False)
    mx_mod.init_params()
    args, auxs = mx_mod.get_params()
    Batch = namedtuple('Batch', ['data'])
    data = mx.nd.array(np.random.uniform(size=image_shape).astype('float32'), ctx=ctx)
    batch = Batch([data])
    return [mx_mod, batch]


def cnn_trial(mod, batch):
    mod.forward(batch)
    return mod.get_outputs()[0].asnumpy()


def cnn_teardown(mod, batch):
    pass
