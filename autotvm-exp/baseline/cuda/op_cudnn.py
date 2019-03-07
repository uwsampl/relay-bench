import time
import argparse

import numpy as np
import tvm
from tvm import autotvm
import topi
from topi.util import get_const_tuple

from util import log_value_old as log_value, array2str_round

raw_conv2d_wkls = [
    # resnet18
    ("resnet.C1.B1",  1,  224, 3,   64,  7, 2, 3, "float32"),
    ("resnet.C2.B1",  1,  56,  64,  64,  3, 1, 1, "float32"),
    ("resnet.C3.B1",  1,  56,  64,  64,  1, 1, 0, "float32"),
    ("resnet.C4.B1",  1,  56,  64,  128, 3, 2, 1, "float32"),
    ("resnet.C5.B1",  1,  56,  64,  128, 1, 2, 0, "float32"),
    ("resnet.C6.B1",  1,  28,  128, 128, 3, 1, 1, "float32"),
    ("resnet.C7.B1",  1,  28,  128, 256, 3, 2, 1, "float32"),
    ("resnet.C8.B1",  1,  28,  128, 256, 1, 2, 0, "float32"),
    ("resnet.C9.B1",  1,  14,  256, 256, 3, 1, 1, "float32"),
    ("resnet.C10.B1", 1,  14,  256, 512, 3, 2, 1, "float32"),
    ("resnet.C11.B1", 1,  14,  256, 512, 1, 2, 0, "float32"),
    ("resnet.C12.B1", 1,  7,   512, 512, 3, 1, 1, "float32"),
]


def test_conv2d_nchw(target, target_host, ctx,
                     batch, in_size, in_channel, out_channel,
                     kernel, stride, padding, dtype,
                     n_times):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), dtype=dtype, name='data')
    W = tvm.placeholder((out_channel, in_channel, kernel, kernel), dtype=dtype, name='weight')

    with tvm.target.create(target):
        B = topi.nn.conv2d(A, W, (stride, stride), (padding, padding), 'NCHW', 'float32')
        s = topi.generic.schedule_conv2d_nchw([B])
        func = tvm.build(s, [A, W, B], target_host=target_host)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    a = tvm.nd.array(np.random.randn(*a_shape).astype(dtype), ctx)
    w = tvm.nd.array(np.random.randn(*w_shape).astype(dtype), ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape)).astype(B.dtype), ctx)

    time_f = func.time_evaluator(func.entry_name, ctx, number=n_times, repeat=3)
    cost = np.mean(time_f(a, w, b).results)

    flop = 2 * np.prod(b.shape) * in_channel * kernel * kernel

    return flop, cost

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='conv2d')
    parser.add_argument("--n-ave-curve", type=int, default=3)
    parser.add_argument("--n-trial", type=int)
    args = parser.parse_args()

    min_repeat_ms = 150

    conv2d_wkls = []
    for wkl in raw_conv2d_wkls:
        for batch_size in [1]:    # [2, 4, 6, 8]
            tmp = list(wkl)
            tmp[0] = tmp[0].replace("B1", "B%d" % batch_size)
            tmp[1] = batch_size
            conv2d_wkls.append(tmp)

    if args.task == 'conv2d':
        for wkl in conv2d_wkls:
            wkl_name, batch, in_size, in_channel, out_channel,\
                    kernel, stride, padding, dtype = wkl

            if batch_size <= 16:
                n_times = 200
            elif batch_size <= 32:
                n_times = 150
            elif batch_size <= 64:
                n_times = 100
            else:
                n_times = 50

            while True:
                costs = []
                for i in range(args.n_ave_curve):
                    flop, cost = test_conv2d_nchw('cuda -libs=cudnn', 'llvm', tvm.gpu(0),
                                                   batch, in_size, in_channel, out_channel,
                                                   kernel, stride, padding, dtype, n_times=n_times)
                    time.sleep(2)
                    costs.append(cost)

                if cost * n_times < min_repeat_ms / 1000.0:
                    n_times = int(min_repeat_ms * 1.1 / 1000 / cost)
                    print("increasing to %d" % n_times)
                    continue

                if np.std(costs) / np.mean(costs) < 0.03:
                    break
                else:
                    print("retry due to high variance ...")

            device_name = tvm.gpu(0).device_name
            log_value('cuda', device_name, wkl_name, 'cudnn', array2str_round(costs))
            cost = np.mean(costs)
            with open("gflops.log", "a") as gflops_file:
                gflops_file.write("\t".join([wkl_name, device_name, array2str_round(flop/1e9/cost)]) + "\n")

