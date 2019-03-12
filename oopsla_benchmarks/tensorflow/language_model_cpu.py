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
from util import measure_rnn_cell, log_value, array2str_round, no_visible_gpus

wkls = [
    ('RNN.B4.L2.S1.H650.V0',      'rnn',  4, 2, 1, 650, 0),
    ('RNN.B4.L2.S1.H650.V10000',  'rnn',  4, 2, 1, 650, 10000),

    ('LSTM.B4.L2.S1.H650.V0',     'lstm', 4, 2, 1, 650, 0),
    ('LSTM.B4.L2.S1.H650.V10000', 'lstm', 4, 2, 1, 650, 10000),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-ave-curve', type=int, default=5)
    parser.add_argument("--n-times", type=int, default=1000,
                                     help="number of runs to take average for time cost")
    parser.add_argument("--dtype", type=str, default='float32')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    with no_visible_gpus():
        for wkl in wkls:
            for xla in [True, False]:
                task_name = wkl[0]
                while True:
                    costs = []
                    for i in range(args.n_ave_curve):
                        tvm_res = measure_rnn_cell(wkl, '/cpu:0', xla, args.n_times)
                        costs.append(tvm_res)

                    if np.std(costs) / np.mean(costs) < 0.03:
                        break
                    print(array2str_round(costs), "retry due to high variance of measure results...")

                print(wkl, array2str_round(costs))
                device_name = tvm.cpu(0).device_name
                method = 'tf-xla' if xla else 'tf'
                log_value(device_name, 'cpu', task_name, wkl, method, '',
                          array2str_round(costs))

