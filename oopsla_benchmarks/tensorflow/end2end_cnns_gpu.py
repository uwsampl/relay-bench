import tensorflow as tf
import numpy as np
import argparse
import time

import tvm

from util import score_cnn, log_value, array2str_round

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ave-curve", type=int, default=3)
    args = parser.parse_args()

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16', 'dcgan']
    dev = '/gpu:0'

    batch_sizes = [1]
    data_format = 'channels_first'

    for net in networks:
        for b in batch_sizes:
            for xla in [False, True]:
                num_batches = 1000 if b == 1 else 100

                while True:
                    costs = []
                    for t in range(args.n_ave_curve):
                        speed = score_cnn(network=net, data_format=data_format, dev=dev,
                                          batch_size=b, num_batches=num_batches, enable_xla=xla)

                        if t != args.n_ave_curve - 1:
                            time.sleep(4)
                        costs.append(1 / speed)

                    if np.std(costs) / np.mean(costs) < 0.04:
                        break
                    print(costs, 'retry due to high variance in measure results')

                method = 'tf-xla' if xla else 'tf'
                device_name = tvm.gpu(0).device_name

                task_name = "%s.B%d" % (net, b)
                log_value(device_name, 'cuda', task_name, net, method, '',
                          array2str_round(costs))
                print(task_name, method, ["%.6f" % x for x in costs])
