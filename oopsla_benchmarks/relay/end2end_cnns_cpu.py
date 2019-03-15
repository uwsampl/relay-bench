import numpy as np
import argparse
import time
import tvm

from oopsla_benchmarks.util import run_experiments
from util import score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ave-curve", type=int, default=3)
    args = parser.parse_args()

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16', 'dcgan']
    batch_sizes = [1]
    num_batches = 1

    device = 'cpu'
    device_name = tvm.cpu(0).device_name

    opt_levels = [3]

    run_experiments(score, args.n_ave_curve,
                    'relay', 'cnn', device_name,
                    ['network', 'device', 'batch_size', 'num_batches', 'opt_level'],
                    [networks, [device], batch_sizes, [num_batches], opt_levels])
