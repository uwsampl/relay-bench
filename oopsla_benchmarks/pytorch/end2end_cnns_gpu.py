import numpy as np
import torch
import argparse
import time
import tvm

import torchvision.models as models

from util import score, array2str_round, log_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ave-curve", type=int, default=3)
    args = parser.parse_args()

    networks = ['resnet-18', 'vgg-16']
    batch_sizes = [1]
    device = 'gpu'

    for net in networks:
        for b in batch_sizes:
            num_batches = 1000 if b == 1 else 100

            while True:
                costs = []
                for t in range(args.n_ave_curve):
                    speed = score(net, device, b, num_batches)

                    if t != args.n_ave_curve - 1:
                        time.sleep(4)
                    costs.append(1 / speed)

                if np.std(costs) / np.mean(costs) < 0.04:
                    break
                print (costs, 'retry due to high variance in measure results')

            method = 'pytorch'
            device_name = tvm.cpu(0).device_name

            task_name = "%s.%d" % (net, b)
            log_value(device_name, 'gpu', task_name, net, method, '', array2str_round(costs))
            print(task_name, method, ["%.6f" % x for x in costs])
