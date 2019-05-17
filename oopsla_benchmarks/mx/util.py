import os
import time
import tvm
import mxnet as mx
import numpy as np
from collections import namedtuple

from mxnet import gluon
from mxnet.gluon.model_zoo import vision

from oopsla_benchmarks.mx.models import mxnet_zoo

#from oopsla_benchmarks.tvm_relay.rnn.bert.static_bert import net as bert

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


def import_gluon_rnn(name):
    if name not in ['rnn', 'gru', 'lstm']:
        raise Exception('Unsupported network: %s' % name)
    data_shape = (5, 1, 128)
    state_shape = (1, 128)
    num_states = 2 if name == 'lstm' else 1

    data_names = ['data%s' % i for i in range(num_states + 1)]
    shapes = {'data' : data_shape}
    for i in range(num_states):
        shapes['state%s' % i] = state_shape

    fname = '%s_i128_h128' % name
    model_data_dir = os.path.dirname(os.path.realpath(__file__))
    net = gluon.nn.SymbolBlock.imports(
        "%s/models/model_zoo_data/%s-symbol.json.data" % (model_data_dir, fname),
        data_names,
        "%s/models/model_zoo_data/%s-0001.params.data" % (model_data_dir, fname))
    return net, num_states, shapes


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


def rnn_setup(network, dev):
    use_gpu = (dev == 'gpu')
    context = mx.gpu(0) if use_gpu else mx.cpu(0)

    net, num_states, shapes = import_gluon_rnn(network)
    net.initialize(ctx=context)
    net.hybridize()

    shape_list = [shapes['data']] + [shapes['state%s' % i] for i in range(num_states)]
    mx_inputs = [mx.nd.array(np.random.rand(*shape).astype('float32'), ctx=context)
                 for shape in shape_list]
    thunk = lambda: net(*mx_inputs)[0].asnumpy()
    return [thunk]


def rnn_trial(thunk):
    thunk()


def rnn_teardown(thunk):
    pass


# def bert_setup(network, dev):
#     ctx = mx.gpu(0) #if dev == 'gpu' else mx.cpu()

#     data1 = mx.nd.array(np.random.uniform(size=(24, 384)).astype('float32'), ctx=ctx)
#     data2 = mx.nd.array(np.random.uniform(size=(24, 384)).astype('float32'), ctx=ctx)
#     data3 = mx.nd.array(np.random.uniform(size=(24,)).astype('float32'), ctx=ctx)
#     bert.initialize(ctx=ctx, force_reinit=True)
#     def x():
#         ret = bert(data1, data2, data3)
#         ret.asnumpy()
#     return [x]


# def bert_trial(thunk):
#     thunk()


# def bert_teardown(thunk):
#     pass
