import os
import mxnet as mx
import numpy as np
from collections import namedtuple

from mxnet import gluon
from mxnet.gluon.model_zoo import vision

from mx_models import mxnet_zoo

from validate_config import validate
from exp_templates import (common_trial_params, common_early_exit, run_template)

def get_network(name, ctx):
    image_shape = (1, 3, 224, 224)
    is_gluon = False
    if 'vgg' in name:
        net = mxnet_zoo.mx_vgg(16)
    elif 'resnet' in name:
        net = mxnet_zoo.mx_resnet(18)
    elif 'dcgan' in name:
        net = mxnet_zoo.mx_dcgan()
    elif 'nature-dqn' in name:
        net = mxnet_zoo.mx_dqn()
    elif 'mobilenet' in name:
        net = vision.get_mobilenet_v2(1.0, pretrained=True, ctx=ctx)
        is_gluon = True
    else:
        raise ValueError("Unsupported network: " + name)

    return net, image_shape, is_gluon


def cnn_setup(network, dev, batch_size):
    ctx = mx.gpu(0) if dev == 'gpu' else mx.cpu()

    net, image_shape, is_gluon = get_network(network, ctx)
    data = mx.nd.array(np.random.uniform(size=image_shape).astype('float32'), ctx=ctx)

    # gluon and non-gluon networks are executed slightly differently
    if is_gluon:
        net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
        net_sym = mx.gluon.nn.SymbolBlock(outputs=net(mx.sym.var('data')),
                                          inputs=mx.sym.var('data'),
                                          params=net.collect_params())
        thunk = lambda: net_sym(data).asnumpy()
        return [thunk]

    mx_mod = mx.mod.Module(net, label_names=None, context=ctx)
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


if __name__ == '__main__':
    run_template(validate_config=validate,
                 check_early_exit=common_early_exit({'frameworks': 'mxnet'}),
                 gen_trial_params=common_trial_params(
                     'mxnet', 'cnn_comp',
                     cnn_trial, cnn_setup, cnn_teardown,
                     ['network', 'device', 'batch_size'],
                     ['networks', 'devices', 'batch_sizes']))
