import tensorflow as tf
import numpy as np
import argparse
import time
import tvm

from oopsla_benchmarks.util import run_experiments
from util import score_cnn, no_visible_gpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ave-curve", type=int, default=3)
    args = parser.parse_args()

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16', 'dcgan']
    dev = '/gpu:0'

    batch_sizes = [1]
    num_batches = 1

    data_format = 'channels_first'
    device_name = tvm.gpu(0).device_name

    run_experiments(score_cnn, args.n_ave_curve,
                    'tf', 'cnn', device_name,
                    ['network', 'data_format', 'device',
                     'batch_size', 'num_batches', 'enable_xla'],
                    [networks,
                     [data_format],
                     [dev],
                     batch_sizes,
                     [num_batches],
                     [False, True]])
