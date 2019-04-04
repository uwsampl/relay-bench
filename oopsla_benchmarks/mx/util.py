import time
import tvm
import mxnet as mx
import numpy as np
from collections import namedtuple

from oopsla_benchmarks.mx.models import mxnet_zoo

from mxnet.gluon.model_zoo import vision

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
    ctx = mx.gpu(0) if dev == 'gpu' else mx.cpu()

    # using gluon for mobilenet, which follows different execution path
    if 'mobilenet' in network:
        image_shape = (1, 3, 224, 224)
        data = mx.nd.array(np.random.uniform(size=image_shape).astype('float32'), ctx=ctx)
        net = vision.get_mobilenet_v2(1.0, pretrained=True, ctx=ctx)
        net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        net_sym = mx.gluon.nn.SymbolBlock(outputs=net(mx.sym.var('data')),
                                          inputs=mx.sym.var('data'),
                                          params=net.collect_params())
        thunk = lambda: net_sym(data).asnumpy()
        return [thunk]

    mx_sym, image_shape = get_network(network)
    data = mx.nd.array(np.random.uniform(size=image_shape).astype('float32'), ctx=ctx)
    mx_mod = mx.mod.Module(mx_sym, label_names=None, context=ctx)
    mx_mod.bind(data_shapes=[('data', image_shape)], for_training=False)
    mx_mod.init_params()
    args, auxs = mx_mod.get_params()
    Batch = namedtuple('Batch', ['data'])
    batch = Batch([data])
    def thunk():
        mx_mod.forward(batch)
        return mx_mod.get_outputs()[0].asnumpy()
    return [thunk]


def cnn_trial(thunk):
    thunk()

def cnn_teardown(thunk):
    pass
