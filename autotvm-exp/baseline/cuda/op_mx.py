"""
generate mxnet kernel baseline for conv2d and depthwise conv2d
"""

import time
import argparse
import logging
import sys
import os

import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

import tvm
from util import log_value, array2str_round

workloads = [
    ("mobilenet.D1.B1", 1, 112, 32, 1, 3, 1, 1, "float32"),
    ("mobilenet.D2.B1", 1, 112, 64, 1, 3, 2, 1, "float32"),
    ("mobilenet.D3.B1", 1, 56, 128, 1, 3, 1, 1, "float32"),
    ("mobilenet.D4.B1", 1, 56, 128, 1, 3, 2, 1, "float32"),
    ("mobilenet.D5.B1", 1, 28, 256, 1, 3, 1, 1, "float32"),
    ("mobilenet.D6.B1", 1, 28, 256, 1, 3, 2, 1, "float32"),
    ("mobilenet.D7.B1", 1, 14, 512, 1, 3, 1, 1, "float32"),
    ("mobilenet.D8.B1", 1, 14, 512, 1, 3, 2, 1, "float32"),
    ("mobilenet.D9.B1", 1, 7, 1024, 1, 3, 1, 1, "float32"),

#    ("resnet.C1.B1",  1, 224, 3,   64,  7, 2, 3, "float32"),
#    ("resnet.C2.B1",  1, 56,  64,  64,  3, 1, 1, "float32"),
#    ("resnet.C3.B1",  1, 56,  64,  64,  1, 1, 0, "float32"),
#    ("resnet.C4.B1",  1, 56,  64,  128, 3, 2, 1, "float32"),
#    ("resnet.C5.B1",  1, 56,  64,  128, 1, 2, 0, "float32"),
#    ("resnet.C6.B1",  1, 28,  128, 128, 3, 1, 1, "float32"),
#    ("resnet.C7.B1",  1, 28,  128, 256, 3, 2, 1, "float32"),
#    ("resnet.C8.B1",  1, 28,  128, 256, 1, 2, 0, "float32"),
#    ("resnet.C9.B1",  1, 14,  256, 256, 3, 1, 1, "float32"),
#    ("resnet.C10.B1", 1, 14,  256, 512, 3, 2, 1, "float32"),
#    ("resnet.C11.B1", 1, 14,  256, 512, 1, 2, 0, "float32"),
#    ("resnet.C12.B1", 1, 7,   512, 512, 3, 1, 1, "float32"),

#    ("mobilenet.D1.B32", 32, 112, 32, 1, 3, 1, 1, "float32"),
#    ("mobilenet.D2.B32", 32, 112, 64, 1, 3, 2, 1, "float32"),
#    ("mobilenet.D3.B32", 32, 56, 128, 1, 3, 1, 1, "float32"),
#    ("mobilenet.D4.B32", 32, 56, 128, 1, 3, 2, 1, "float32"),
#    ("mobilenet.D5.B32", 32, 28, 256, 1, 3, 1, 1, "float32"),
#    ("mobilenet.D6.B32", 32, 28, 256, 1, 3, 2, 1, "float32"),
#    ("mobilenet.D7.B32", 32, 14, 512, 1, 3, 1, 1, "float32"),
#    ("mobilenet.D8.B32", 32, 14, 512, 1, 3, 2, 1, "float32"),
#    ("mobilenet.D9.B32", 32, 7, 1024, 1, 3, 1, 1, "float32"),
#
#    ("resnet.C1.B32",  32, 224, 3,   64,  7, 2, 3, "float32"),
#    ("resnet.C2.B32",  32, 56,  64,  64,  3, 1, 1, "float32"),
#    ("resnet.C3.B32",  32, 56,  64,  64,  1, 1, 0, "float32"),
#    ("resnet.C4.B32",  32, 56,  64,  128, 3, 2, 1, "float32"),
#    ("resnet.C5.B32",  32, 56,  64,  128, 1, 2, 0, "float32"),
#    ("resnet.C6.B32",  32, 28,  128, 128, 3, 1, 1, "float32"),
#    ("resnet.C7.B32",  32, 28,  128, 256, 3, 2, 1, "float32"),
#    ("resnet.C8.B32",  32, 28,  128, 256, 1, 2, 0, "float32"),
#    ("resnet.C9.B32",  32, 14,  256, 256, 3, 1, 1, "float32"),
#    ("resnet.C10.B32", 32, 14,  256, 512, 3, 2, 1, "float32"),
#    ("resnet.C11.B32", 32, 14,  256, 512, 1, 2, 0, "float32"),
#    ("resnet.C12.B32", 32, 7,   512, 512, 3, 1, 1, "float32"),
]

def measure_conv(wkl, n_times):
    name, N, H, CI, CO, HK, HSTR, HPAD, dtype = wkl
    W = H
    WK = HK
    WSTR = HSTR
    WPAD = HPAD

    OH = (H + 2*HPAD - HK) // HSTR + 1
    OW = (W + 2*WPAD - WK) // WSTR + 1

    dshape = (N, CI, H, W)
    kshape = (CO, CI, HK, WK)
    oshape = (N, CO, OH, OW)

    ctx = mx.gpu(0)
    na = nd.array(np.random.uniform(size=dshape).astype(dtype), ctx)
    nb = nd.array(np.random.uniform(size=kshape).astype(dtype), ctx)

    for i in range(10):
        out = nd.Convolution(na, nb, num_filter=CO, kernel=(HK, WK),
                             pad=(HPAD, WPAD), stride=(HSTR, WSTR), no_bias=True)
    assert out.shape == oshape
    out.wait_to_read()

    tic = time.time()
    for i in range(n_times):
        out = nd.Convolution(na, nb, num_filter=CO, kernel=(HK, WK),
                             pad=(HPAD, WPAD), stride=(HSTR, WSTR), no_bias=True)
    out.wait_to_read()
    cost = time.time() - tic

    return cost / n_times

def measure_depthwise_conv(wkl, n_times):
    name, N, H, CI, CM, HK, HSTR, HPAD, dtype = wkl
    W = H
    WK = HK
    WSTR = HSTR
    WPAD = HPAD

    OH = (H + 2*HPAD - HK) // HSTR + 1
    OW = (W + 2*WPAD - WK) // WSTR + 1
    CO = CI * CM

    dshape = (N, CI, H + 2 * HPAD, W + 2 * WPAD)
    kshape = (CI, CM, HK, WK)
    oshape = (N, CO, OH, OW)

    ctx = mx.gpu(0)
    na = nd.array(np.random.uniform(size=dshape).astype(dtype), ctx)
    nb = nd.array(np.random.uniform(size=kshape).astype(dtype), ctx)

    for i in range(10):
        out = nd.Convolution(na, nb, num_filter=CO, num_group=CO,
                             kernel=(HK, WK), pad=(0, 0), stride=(HSTR, WSTR), no_bias=True)
    assert out.shape == oshape
    out.wait_to_read()

    tic = time.time()
    for i in range(n_times):
        out = nd.Convolution(na, nb, num_filter=CO, num_group=CO,
                             kernel=(HK, WK), pad=(0, 0), stride=(HSTR, WSTR), no_bias=True)
    out.wait_to_read()
    cost = time.time() - tic

    return cost / n_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ave-curve", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for wkl in workloads:
        if 'C' in wkl[0]:
            measure_func, n_times = measure_conv, 2000
        elif 'D' in wkl[0]:
            measure_func, n_times = measure_depthwise_conv, 1000

        while True:
            costs = []
            for i in range(args.n_ave_curve + 1):
                if i != 0:
                    time.sleep(1)
                cost = measure_func(wkl, n_times)
                costs.append(cost)
            del costs[costs.index(max(costs))]

            if np.std(costs) / np.mean(costs) < 0.06:
                break
            print(array2str_round(costs), np.std(costs) / np.mean(costs),
                  "retry due to high variance of measure results")

        print(wkl[0], array2str_round(costs))
        device_name = tvm.gpu(0).device_name
        log_value('cuda', device_name, wkl[0], 'mxnet', array2str_round(costs))

