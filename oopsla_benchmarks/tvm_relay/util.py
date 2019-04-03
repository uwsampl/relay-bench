import time
import tvm
from tvm import relay
import numpy as np
import onnx
import keras
import tensorflow
import mxnet as mx
import aot

from oopsla_benchmarks.tvm_relay.rnn import char_rnn_generator as rnn
from oopsla_benchmarks.tvm_relay.rnn import samples
from oopsla_benchmarks.util.language_data import N_LETTERS

# from oopsla_benchmarks.tvm_relay.rnn.bert.static_bert import model as bert

from oopsla_benchmarks.tvm_relay.models import mxnet_zoo, onnx_zoo

def get_network(name, batch_size, dtype='float32', ir='relay'):
    """Get the symbol definition and random weight of a network
    
    Parameters
    ----------
    name: str
        The name of the network, can be 'resnet-18', 'resnet-50', 'vgg-16', 'inception_v3', 'mobilenet', ...
    batch_size: int
        batch size
    dtype: str
        Data type

    Returns
    -------
    net: nnvm.symbol
        The NNVM symbol of network definition
    params: dict
        The random parameters for benchmark
    input_shape: tuple
        The shape of input tensor
    """
    if ir == 'relay':
        from tvm.relay import testing
    elif ir == 'nnvm':
        from nnvm import testing
    else:
        raise Exception("ir must be `relay` or `nnvm`, but you used `{}`".format(ir))

    input_shape = (batch_size, 3, 224, 224)
    if name == 'mobilenet':
        net, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet_v2':
        net, params = testing.mobilenet_v2.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (batch_size, 3, 299, 299)
        net, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "densenet" in name:
        n_layer = int(name.split('-')[1])
        net, params = testing.densenet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "squeezenet" in name:
        version = name.split("_v")[1]
        net, params = testing.squeezenet.get_workload(batch_size=batch_size, version=version, dtype=dtype)
    elif name == 'custom':
        # an example for custom network
        # from tvm.relay.testing import init
        # net = relay.var('data')
        # net = relay.testing.layers.conv2d(net, channels=4, kernel_size=(3,3), padding=(1,1))
        # net = relay.nn.batch_flatten(net)
        # net = relay.testing.layers.dense_add_bias(net, units=1000)
        # net, params = init.create_workload(net, batch_size, (3, 224, 224))
        from tvm.relay.testing import init
        input_shape = (3, 224)
        net = relay.var('data', shape=input_shape)
        weight = relay.var('dense_weight', shape=(224, 224))
        net = relay.nn.dense(net, weight)
        net = relay.Function(relay.ir_pass.free_vars(net), net)
        # net = relay.testing.layers.dense_add_bias(net, name="dense")
        net, params = init.create_workload(net)
    # simple networks for experimenting
    elif name == 'mlp':
        image_shape = (1, 28, 28)
        input_shape = (batch_size,) + image_shape
        net, params = testing.mlp.get_workload(batch_size=batch_size, image_shape=image_shape)
    elif name == 'nature-dqn':
        image_shape = (4, 84, 84)
        input_shape = (batch_size,) + image_shape
        net, params = testing.dqn.get_workload(batch_size=batch_size, image_shape=image_shape)
    elif name == 'dcgan':
        random_len = 100
        input_shape = (batch_size, random_len)
        net, params = testing.dcgan.get_workload(batch_size, random_len=random_len)
    elif name == 'densenet':
        input_shape = (3, 64, 64)
        net, params = testing.densenet.get_workload(batch_size=batch_size)
    # elif name == 'mxnet':
    #     # an example for mxnet model
    #     from mxnet.gluon.model_zoo.vision import get_model
    #     block = get_model('resnet18_v1', pretrained=True)
    #     net, params = nnvm.frontend.from_mxnet(block)
    #     net = nnvm.sym.softmax(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape


def get_mxnet_network(name):
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


def get_onnx_network(name):
    image_shape = (1, 3, 224, 224)
    if 'resnet' in name:
        net = onnx_zoo.resnet18_1_0
    elif 'vgg' in name:
        net = onnx_zoo.vgg16
    elif 'mobilenet' in name:
        net = onnx_zoo.mobilenet2
    else:
        raise ValueError("Unsupported network: " + name)


    return net, image_shape


def get_keras_network(name):
    image_shape = (224, 224, 3)
    if 'vgg' in name:
        model = keras.applications.VGG16(include_top=True, weights='imagenet',
                                         input_shape=image_shape, classes=1000)
    elif 'mobilenet' in name:
        model = keras.applications.MobileNet(include_top=True, weights='imagenet',
                                             input_shape=image_shape, classes=1000)
    else:
        raise ValueError("Unsupported network: " + name)

    return model, image_shape


def setup_relay_mod(net, image_shape, input_name, params, dev, opt):
    device = tvm.cpu(0) if dev == 'cpu' else tvm.gpu(0)
    with relay.build_module.build_config(opt_level=opt):
        graph, lib, params = relay.build(net, 'llvm' if dev == 'cpu' else 'cuda', params=params)

    mod = tvm.contrib.graph_runtime.create(graph, lib, ctx=device)
    mod.set_input(**params)
    mod.set_input(input_name,
                  tvm.nd.array((np.random.uniform(size=image_shape)).astype('float32')))
    return mod


def mxnet_setup(network, dev, batch_size, opt):
    mx_sym, image_shape = get_mxnet_network(network)
    mx_mod = mx.mod.Module(mx_sym, label_names=None)
    mx_mod.bind(data_shapes=[('data', image_shape)], for_training=False)
    mx_mod.init_params()
    args, auxs = mx_mod.get_params()

    new_sym, params = relay.frontend.from_mxnet(mx_sym, {'data': image_shape}, arg_params=args, aux_params=auxs)
    mod = setup_relay_mod(new_sym, image_shape, 'data', params, dev, opt)
    return [mod]


def onnx_setup(network, dev, batch_size, opt):
    net, image_shape = get_onnx_network(network)
    model = onnx.load_model(net)
    sym, params = relay.frontend.from_onnx(model, {model.graph.input[0].name: image_shape})
    mod = setup_relay_mod(sym, image_shape, model.graph.input[0].name, params, dev, opt)
    return [mod]


def keras_setup(network, dev, batch_size, opt):
    model, image_shape = get_keras_network(network)
    func, params = relay.frontend.from_keras(model, {model.input_names[0]: image_shape})
    mod = setup_relay_mod(func, image_shape, model.input_names[0], params, dev, opt)
    return [mod]


def cnn_setup(network, dev, batch_size, opt):
    net, params, image_shape = get_network(network, batch_size)
    mod = setup_relay_mod(net, image_shape, 'data', params, dev, opt)
    return [mod]


def cnn_trial(mod):
    return mod.run()


def cnn_teardown(mod):
    pass


def rnn_setup(network, device, configuration, method, hidden_size, lang, letters):
    if network != 'char-rnn':
        raise Exception('Only supported network is char-rnn')
    gpu = (device == 'gpu')
    cell_only = (configuration == 'cell')
    aot = (method == 'aot')

    net = rnn.RNNCellOnly if cell_only else rnn.RNNLoop
    init_net = net(aot, gpu, N_LETTERS, hidden_size, N_LETTERS)
    if cell_only:
        thunk = lambda: samples(init_net, lang, letters)
    else:
        thunk = lambda: init_net.samples(lang, letters)
    return [thunk]


def rnn_trial(thunk):
    return thunk()


def rnn_teardown(thunk):
    pass


# def bert_setup(network, dev, method):
#     net, params = bert
#     gpu = (dev == 'gpu')
#     use_aot = (method == 'aot')
#     device = tvm.gpu(0) if gpu else tvm.cpu(0)

#     if not use_aot:
#         with relay.build_module.build_config(opt_level=1):
#             graph, lib, params = relay.build(net, 'cuda' if gpu else 'llvm', params=params)

#             mod = tvm.contrib.graph_runtime.create(graph, lib, ctx=device)
#             mod.set_input(**params)
#             mod.set_input('data0',
#                           tvm.nd.array((np.random.uniform(size=(24, 384))).astype('float32')))
#             mod.set_input('data1',
#                           tvm.nd.array((np.random.uniform(size=(24, 384))).astype('float32')))
#             mod.set_input('data2',
#                           tvm.nd.array((np.random.uniform(size=(24,))).astype('float32')))
#             thunk = lambda: mod.run()
#             return [thunk]
#     else:
#         mod = relay.Module()
#         forward_var = relay.GlobalVar('forward_var')
#         target = tvm.target.cuda() if gpu else tvm.target.create('llvm')
#         forward = aot.compile(mod, forward_var, ctx=device, tgt=target, use_gpu=gpu)
#         args = [relay.Constant(tvm.nd.array((np.random.uniform(size=(24, 384))).astype('float32'))),
#                 relay.Constant(tvm.nd.array((np.random.uniform(size=(24, 384))).astype('float32'))),
#                 relay.Constant(tvm.nd.array((np.random.uniform(size=(24,))).astype('float32')))]
#         aot_args = [aot.convert(a, device) for a in args]
#         thunk = lambda: forward(*aot_args)


# def bert_trial(thunk):
#     return thunk()


# def bert_teardown(mod):
#     pass
