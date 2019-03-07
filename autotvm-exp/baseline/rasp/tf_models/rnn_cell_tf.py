"""
Benchmark single layer single cell performance
unroll 30 times
"""

from collections import namedtuple
import argparse
import logging
import time

import numpy as np

import tensorflow as tf

wkls = [
    ('RNN.B4.L2.S1.H650.V0',      'rnn',  4, 2, 1, 650, 0),
    ('RNN.B4.L2.S1.H650.V10000',  'rnn',  4, 2, 1, 650, 10000),

    ('LSTM.B4.L2.S1.H650.V0',     'lstm', 4, 2, 1, 650, 0),
    ('LSTM.B4.L2.S1.H650.V10000', 'lstm', 4, 2, 1, 650, 10000),
]

def measure_cell(cell_wkl):
    task_name, cell_type, batch_size, num_layer, seq_len, hidden_size, voc_size = cell_wkl

    if cell_type == 'rnn':
        cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    elif cell_type == 'lstm':
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size, hidden_size, state_is_tuple=True)
    elif cell_type == 'basic_lstm':
        cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True)
    else:
        raise Exception('Unknown network! '+network_type)

    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layer, state_is_tuple=True)

    with tf.variable_scope(task_name, reuse=tf.AUTO_REUSE):
        if voc_size:
            # encoder
            W = tf.get_variable('W', [voc_size, hidden_size])
            #ids = [tf.constant(np.random.randint(0, voc_size, [batch_size]).astype(np.int32)) for _ in range(seq_len)]
            ids = [tf.placeholder(tf.int32, shape=(batch_size)) for _ in range(seq_len)]
            data = [tf.nn.embedding_lookup(W, x) for x in ids]
        else:
            #data = [tf.constant(np.random.randn(batch_size, hidden_size).astype(np.float32)) for _ in range(seq_len)]
            data = [tf.placeholder(tf.float32, shape=(batch_size, hidden_size)) for _ in range(seq_len)]

        output, _cell_state = tf.nn.static_rnn(cell, data, initial_state=init_state)

        if voc_size:
            with tf.variable_scope('softmax'):
                W = tf.get_variable('W', [hidden_size, voc_size])
                b = tf.get_variable('b', [voc_size])
            output = [tf.matmul(x, W) + b for x in output]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if voc_size:
                inputs = {'x' + str(item[0]):item[1] for item in enumerate(ids)}
            else:
                inputs = {'x' + str(item[0]):item[1] for item in enumerate(data)}
            outputs = {'y' + str(item[0]):item[1] for item in enumerate(output)}
            tf.saved_model.simple_save(sess,
            './' + task_name.replace('.', '_') + '_model',
            inputs=inputs, outputs=outputs)
            print("saving", task_name + '_model', "...")

if __name__ == '__main__':
    for wkl in wkls:
        measure_cell(wkl)
