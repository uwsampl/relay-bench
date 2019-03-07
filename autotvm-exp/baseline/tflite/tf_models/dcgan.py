""" copied from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py """

import tensorflow as tf

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
    if with_w:
        return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
        return tf.matmul(input_, matrix) + bias

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=False):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum, 
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            fused=True)

def deconv2d(input_, output_shape,
             k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
    
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
        return deconv, w, biases
    else:
        return deconv
 

def dcgan(inputs, oshape, batch_size=1, ngf=128, is_training=False, scope=None):
    assert is_training == False
    assert oshape[-2] == 32
    assert oshape[-3] == 32

    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')

    with tf.variable_scope(scope):
        z_  = linear(inputs, ngf*4*4*4, 'g_h0_lin')

        h0 = tf.reshape(z_, [batch_size, 4, 4, ngf * 4])

        h1 = deconv2d(h0, [batch_size, 8, 8, ngf*2], k_h=4, k_w=4, name='g_h1')
        h1 = tf.nn.relu(g_bn1(h1))

        h2 = deconv2d(h1, [batch_size, 16, 16, ngf], k_h=4, k_w=4, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))

        h3 = deconv2d(h2, (batch_size,) + oshape, k_h=4, k_w=4, name='g_h3')
        net = tf.nn.tanh(h3)

    return net

