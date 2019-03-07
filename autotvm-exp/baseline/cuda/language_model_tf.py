"""
Benchmark single rnn cell and language model (embedding + rnn + decoder)
"""

from collections import namedtuple
import argparse
import logging
import time

import numpy as np

import tensorflow as tf

import tvm
from util import log_value, array2str_round

wkls = [
#    ('RNN.B4.L2.S1.H650.V0',      'rnn',  4, 2, 1, 650, 0),
#    ('RNN.B4.L2.S1.H650.V10000',  'rnn',  4, 2, 1, 650, 10000),

    ('LSTM.B4.L2.S1.H650.V0',     'lstm', 4, 2, 1, 650, 0),
    ('LSTM.B4.L2.S1.H650.V10000', 'lstm', 4, 2, 1, 650, 10000),
]

def measure_cell(cell_wkl, xla, n_times):
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
        with tf.device('/gpu:0'):
            init_state = cell.zero_state(batch_size, tf.float32)
            if voc_size:
                # encoder
                W = tf.get_variable('W', [voc_size, hidden_size])
                ids = [tf.constant(np.random.randint(0, voc_size, [batch_size]).astype(np.int32)) for _ in range(seq_len)]
                data = [tf.nn.embedding_lookup(W, x) for x in ids]
            else:
                data = [tf.constant(np.random.randn(batch_size, hidden_size).astype(np.float32)) for _ in range(seq_len)]

            output, _cell_state = tf.nn.static_rnn(cell, data, initial_state=init_state)

            if voc_size:
                with tf.variable_scope('softmax'):
                    W = tf.get_variable('W', [hidden_size, voc_size])
                    b = tf.get_variable('b', [voc_size])
                output = [tf.matmul(x, W) + b for x in output]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    config = tf.ConfigProto(log_device_placement=False)
    if xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dry_run = 5
        for i in range(dry_run + n_times):
            if i == dry_run:
                tic = time.time()
            out = sess.run(output)
        end = time.time()

    return (end - tic) / n_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-ave-curve', type=int, default=5)
    parser.add_argument("--n-times", type=int, default=1000,
                                     help="number of runs to take average for time cost")
    parser.add_argument("--dtype", type=str, default='float32')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)


    for wkl in wkls:
        for xla in [True, False]:
            task_name = wkl[0]
            while True:
                costs = []
                for i in range(args.n_ave_curve):
                    tvm_res = measure_cell(wkl, xla, args.n_times)
                    costs.append(tvm_res)

                if np.std(costs) / np.mean(costs) < 0.03:
                    break
                print(array2str_round(costs), "retry due to high variance of measure results...")

            print(wkl, array2str_round(costs))
            device_name = tvm.gpu(0).device_name
            method = 'tf-xla' if xla else 'tf'
            log_value('cuda', device_name, task_name, method,
                      array2str_round(costs))

