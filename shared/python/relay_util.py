import tvm
from tvm import relay
import numpy as np
import aot

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
        net = relay.Function(relay.analysis.free_vars(net), net)
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


def setup_relay_mod(net, image_shape, input_name, params, dev, opt):
    device = tvm.cpu(0) if dev == 'cpu' else tvm.gpu(0)
    with relay.build_config(opt_level=opt):
        graph, lib, params = relay.build(net, 'llvm' if dev == 'cpu' else 'cuda', params=params)

    mod = tvm.contrib.graph_runtime.create(graph, lib, ctx=device)
    mod.set_input(**params)
    mod.set_input(input_name,
                  tvm.nd.array((np.random.uniform(size=image_shape)).astype('float32')))
    return mod


def cnn_setup(network, dev, batch_size, opt):
    net, params, image_shape = get_network(network, batch_size)
    mod = setup_relay_mod(net, image_shape, 'data', params, dev, opt)
    return [mod]


def cnn_trial(mod):
    return mod.run()


def cnn_teardown(mod):
    pass
