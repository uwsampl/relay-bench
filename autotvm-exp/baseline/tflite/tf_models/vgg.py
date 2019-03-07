# copied from https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/README.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def vgg_16(inputs, is_training=False, scope=None):
    assert is_training == False
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            #net = slim.dropout(net, 0.5, scope='dropout6')
            net = slim.fully_connected(net, 4096, scope='fc7')
            #net = slim.dropout(net, 0.5, scope='dropout7')
            net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
            net = slim.softmax(net)
    return net

