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

from oopsla_benchmarks.util import run_experiments
from util import measure_rnn_cell, no_visible_gpus

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

    dev = '/cpu:0'
    device_name = tvm.cpu(0).device_name

    with no_visible_gpus():
        run_experiments(measure_rnn_cell, args.n_ave_curve,
                        'tf', 'rnn', device_name,
                        ['workload', 'device', 'enable_xla', 'n_times'],
                        [wkls, [dev], [True, False], [args.n_times]])
