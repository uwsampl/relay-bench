import tensorflow as tf
import numpy as np
import argparse
import time

from tf_models.resnet import *
from tf_models.vgg import *
from tf_models.mobilenet import *
from tf_models.dqn import *
from tf_models.dcgan import *

import tvm

from util import log_value, array2str_round

def score(network, dev, batch_size, num_batches, enable_xla):
    data_format = 'channels_first'

    image_shape = (batch_size, 224, 224, 3)

    if network == 'resnet-18':
        net = imagenet_resnet_v2(resnet_size=18, num_classes=1000, data_format=data_format)
    elif network == 'mobilenet':
        net = wrapped_partial(mobilenet_v1, depth_multiplier=1.0,
                              scope="%d" % (int(np.random.randint(1 << 31))))
    elif network == 'vgg-16':
        net = wrapped_partial(vgg_16,
                              scope="%d" % (int(np.random.randint(1 << 31))))
    elif network == 'nature-dqn':
        net = wrapped_partial(nature_dqn,
                              scope="%d" % (int(np.random.randint(1 << 31))))
        image_shape = (batch_size, 84, 84, 4)
    elif network == 'dcgan':
        net = wrapped_partial(dcgan, oshape=(32, 32, 3), batch_size=batch_size,
                              scope="%d" % (int(np.random.randint(1 << 31))))
        image_shape = (batch_size, 100)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ave-curve", type=int, default=3)
    args = parser.parse_args()

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16', 'dcgan']
    # networks = ['dcgan']
    dev = '/gpu:0'

    batch_sizes = [1]

    for net in networks:
        for b in batch_sizes:
            for xla in [False, True]:
                num_batches = 1000 if b == 1 else 100

                while True:
                    costs = []
                    for t in range(args.n_ave_curve):
                        speed = score(network=net, dev=dev, batch_size=b, num_batches=num_batches,
                                      enable_xla=xla)
                        if t != args.n_ave_curve - 1:
                            time.sleep(4)
                        costs.append(1 / speed)

                    if np.std(costs) / np.mean(costs) < 0.04:
                        break
                    print(costs, 'retry due to high variance in measure results')

                method = 'tf-xla' if xla else 'tf'
                device_name = tvm.gpu(0).device_name

                task_name = "%s.B%d" % (net, b)
                log_value(device_name, 'cuda', task_name, net, method, '',
                          array2str_round(costs))
                print(task_name, method, ["%.6f" % x for x in costs])
            exit()

