import argparse
import os
import mxnet as mx
import numpy as np
import sys
from collections import namedtuple

from mxnet import gluon
from mxnet.gluon.model_zoo import vision

from mx_models import mxnet_zoo

from validate_config import validate
from common import write_status
from trial_util import configure_seed, run_trials

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


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'mxnet' not in config['frameworks']:
        write_status(output_dir, True, 'MxNet not run')
        sys.exit(0)

    configure_seed(config)

    success, msg = run_trials(
        'mxnet', 'cnn_comp',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        cnn_trial, cnn_setup, cnn_teardown,
        ['network', 'device', 'batch_size'],
        [config['networks'], config['devices'], config['batch_sizes']],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir)
