# copied from https://github.com/tensorflow/models/blob/master/research/inception/inception/slim/README.md

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def nature_dqn(inputs, is_training=False, scope=None, num_action=18):
    assert is_training == False
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.conv2d(inputs, 32, [8, 8], 4, padding='VALID', scope='conv1')
            net = slim.conv2d(net, 64, [4, 4], 2, padding='VALID', scope='conv2')
            net = slim.conv2d(net, 64, [3, 3], 1, padding='VALID', scope='conv3')
            net = slim.fully_connected(net, 512, scope='fc4')
            net = slim.fully_connected(net, num_action, activation_fn=None, scope='fc5')
    return net

if __name__ == '__main__':
    print("building dqn model...")
    batch_size = 1
    input_height = 84
    input_width = 84
    input_channels = 4
    inp = tf.placeholder(tf.float32, shape=(batch_size, input_height,
                             input_width, input_channels), name='x')
    with tf.Session() as sess:
        output = nature_dqn(inp, is_training=False,
                            scope='{0:d}'.format(int(np.random.randint(1<<31))))
        sess.run(tf.global_variables_initializer())
        tf.saved_model.simple_save(sess, './dqn_model',
        inputs={'x': inp}, outputs={'y': output})
    print("saving dqn model file...")


