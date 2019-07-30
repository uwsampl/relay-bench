import argparse
import os
import numpy as np
import sys
import tensorflow as tf

from validate_config import validate
from common import write_status
from trial_util import run_trials

from tf_models import (mobilenet, resnet, vgg, dqn, dcgan)

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


def cnn_setup(network, device, batch_size, enable_xla):
    dev = '/gpu:0' if device == 'gpu' else '/cpu:0'

    # for CPU, the data format must be channels_last because certain
    # channels_first implementations exist only for GPU
    data_format = 'channels_first' if dev == '/gpu:0' else 'channels_last'
    net, image_shape = instantiate_network(network, batch_size, data_format)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), log_device_placement=False)
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


def main(config_dir, output_dir, device):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'tf' not in config['frameworks']:
        write_status(output_dir, True, 'TF not run')
        sys.exit(0)

    if device not in config['devices']:
        write_status(output_dir, True, 'TF not run on {}'.format(device))
        sys.exit(0)

    enable_xla = [False]
    if config['use_xla']:
        enable_xla.append(True)

    success, msg = run_trials(
        'tf', 'cnn_comp',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        cnn_trial, cnn_setup, cnn_teardown,
        ['network', 'device', 'batch_size', 'enable_xla'],
        [config['networks'], [device],
         config['batch_sizes'], enable_xla],
        path_prefix=output_dir,
        append_to_csv=True)

    write_status(output_dir, success, msg)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir, args.device)
