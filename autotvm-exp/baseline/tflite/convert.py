import os
import argparse

import numpy as np
from scipy.misc import imsave

import tensorflow as tf

from util import devices

from tf_models.resnet import *
from tf_models.vgg import *
from tf_models.mobilenet import *
from tf_models.dqn import *
from tf_models.dcgan import *

def convert(network, batch_size):
    data_format = 'channels_last'

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

    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, shape=image_shape)
        output = net(inputs, is_training=False)

        converter = tf.contrib.lite.TocoConverter.from_session(sess, [inputs], [output])
        tflite_model = converter.convert()
        open(network + ".tflite", "wb").write(tflite_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ave-curve", type=int, default=3)
    args = parser.parse_args()

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']

    # convert model
    for net in networks:
        print("Converting " + net + "...")
        convert(net, batch_size=1)

    # send to devices
    for device in devices:
        cmd = "scp *.tflite " + device.ssh_address + ":autotvm-exp/baseline/tflite"
        print(cmd)
        os.system(cmd)

