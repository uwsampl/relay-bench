"""
generate tensor comprehension baseline for conv2d and depthwise conv2d
Usage: python3 op_tc.py --n-trial 1000
"""

import time
import argparse
import logging
import sys
import os
import re
import queue

import numpy as np
import torch
from scipy.interpolate import interp1d

import tensor_comprehensions as tc
import tvm
from util import log_value, array2str_round

#tc.GlobalDebugInit(["--dump_cuda=true"])

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

    ("resnet.C1.B1",  1, 224, 3,   64,  7, 2, 3, "float32"),
    ("resnet.C2.B1",  1, 56,  64,  64,  3, 1, 1, "float32"),
    ("resnet.C3.B1",  1, 56,  64,  64,  1, 1, 0, "float32"),
    ("resnet.C4.B1",  1, 56,  64,  128, 3, 2, 1, "float32"),
    ("resnet.C5.B1",  1, 56,  64,  128, 1, 2, 0, "float32"),
    ("resnet.C6.B1",  1, 28,  128, 128, 3, 1, 1, "float32"),
    ("resnet.C7.B1",  1, 28,  128, 256, 3, 2, 1, "float32"),
    ("resnet.C8.B1",  1, 28,  128, 256, 1, 2, 0, "float32"),
    ("resnet.C9.B1",  1, 14,  256, 256, 3, 1, 1, "float32"),
    ("resnet.C10.B1", 1, 14,  256, 512, 3, 2, 1, "float32"),
    ("resnet.C11.B1", 1, 14,  256, 512, 1, 2, 0, "float32"),
    ("resnet.C12.B1", 1, 7,   512, 512, 3, 1, 1, "float32"),

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

settings = {
    "threads": 8, "generations": 10, "pop_size": 100, "number_elites": 3
}

conv2d_lang = """
def conv2d(float(N, C, H, W) I, float(M, C, KH, KW) W1) -> (output) {{
    output(n, m, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(m, c, kh, kw)
}}
"""

depthwise_conv2d_lang = """
def depthwise_conv2d(float(N, C, H, W) I, float(C, KH, KW) W1) -> (output) {{
    output(n, c, h, w) +=! I(n, c, {sh} * h + kh, {sw} * w + kw) * W1(c, kh, kw)
}}
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--n-ave-curve", type=int, default=2)
    parser.add_argument("--n-trial", type=int)
    args = parser.parse_args()

    if args.n_trial == 2000:
        settings = {"generations": 20, "pop_size": 100, "number_elites": 3}
    if args.n_trial == 1000:
        settings = {"generations": 10, "pop_size": 100, "number_elites": 3}
    elif args.n_trial == 400:
        settings = {"generations": 8, "pop_size": 50, "number_elites": 2}
    elif args.n_trial == 10:
        settings = {"generations": 5, "pop_size": 2, "number_elites": 0}
    elif args.n_trial == 4:
        settings = {"generations": 2, "pop_size": 2, "number_elites": 0}
    else:
        raise RuntimeError("Invalid number of trial")

    logging.basicConfig(level=logging.INFO)

    if '.D' in args.task or '.C' in args.task:
        # subprocess called by master process
        # this is a walkaround for extracting the stdout of c function called by TC
        items = None
        for wkl in workloads:
            if wkl[0] == args.task:
                items = wkl
        assert items is not None

        if 'C' in items[0]:
            _, N, H, C, M, KH, sh, pad, dtype = items
            H = H + 2 * pad
            W = H
            KW = KH
            sw = sh
            OH = (H - KH) // sw + 1
            OW = (W - KW) // sw + 1

            cache_file = 'cache_%s.tc' % str(wkl)
            conv2d = tc.define(conv2d_lang, name="conv2d", constants={"sh": sh, "sw": sw})

            options = tc.Options('conv')

            data, kernel = torch.randn(N, C, H, W).cuda(), torch.randn(M, C, KH, KW).cuda()
            conv2d.autotune(data, kernel, options=options, **settings)

            out = conv2d(data, kernel)
            assert out.shape == (N, M, OH, OW)
        elif 'D' in items[0]:
            _, N, H, C, M, KH, sh, pad, dtype = items
            H = H + 2 * pad
            W = H
            KW = KH
            sw = sh

            OH = (H - KH) // sw + 1
            OW = (W - KW) // sw + 1
            assert M == 1

            cache_file = 'cache_%s.tc' % str(wkl)
            dw_conv2d = tc.define(depthwise_conv2d_lang, name="depthwise_conv2d",
                                  constants={"sh": sh, "sw": sw})
            options = tc.Options('conv')
            data, kernel = torch.randn(N, C, H, W).cuda(), torch.randn(C, KH, KW).cuda()
            dw_conv2d.autotune(data, kernel, options=options, **settings)
            out = dw_conv2d(data, kernel)
            assert out.shape == (N, C, OH, OW)
    else:
        # master mode
        print("pid for force kill: %d" % os.getpid())

        # tmp log file for extracting stdout of TC's c function
        tmp_filename = 'tc_output.log'
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)

        que = queue.Queue()

        for wkl in workloads:
            que.put(wkl)

        while not que.empty():
            wkl = que.get()

            tic = time.time()
            if '.C' in wkl[0]:
                wkl_name, N, H, C, M, KH, sh, pad, dtype = wkl
                KW = KH
                sw = sh
                OH = (H + 2 * pad - KH) // sh + 1
                OW = OH
                flop = 2 * N * OH * OW * C * M * KH * KW
            elif '.D' in wkl[0]:
                wkl_name, N, H, C, M, KH, sh, pad, dtype = wkl
                KW = KH
                sw = sh
                OH = (H + 2 * pad - KH) // sh + 1
                OW = OH
                flop = 2 * N * OH * OW * C * M * KH * KW

            curves = []
            costs = []
            gflops = []
            error_occured = False
            for j in range(args.n_ave_curve):
                cmd = 'python3 op_tc.py --task "%s" --n-trial %d >> %s 2>> %s' \
                        % (wkl_name, args.n_trial, tmp_filename, tmp_filename)

                print(cmd)
                ret = os.system(cmd)
                if ret != 0:
                    error_occured = True

                xs = []
                ys = []
                with open(tmp_filename) as f:
                    lines = list(f.readlines())
                    # trackback to extract best for every generation
                    best = 1e9
                    for line in reversed(lines):
                        if 'Generation' in line:
                            find = re.search("Generation\s+(\d+).+?, (\d+)\)/.+?us:\s+(\d+)",
                                             line)
                            if find is not None:
                                gen, no, cost = [int(x) for x in
                                                 (find.group(1), find.group(2), find.group(3))]
                                no = gen * settings['pop_size'] + no
                                if not xs or xs[-1] != no:
                                    xs.append(no)
                                    ys.append(flop / (cost / 1e6) / 1e9)
                        if 'Autotuning' in line:
                            break
                    cost = best / 1e6 # us -> s
                xs = [0] + list(reversed(xs))
                ys = [0] + list(reversed(ys))

                for j in range(settings['generations']):
                    node = (j + 1) * settings['pop_size']
                    if node not in xs:
                        print("! Error, cannot found node %d" % node)
                        error_occured = True

                if error_occured:
                    continue

                keep = 0
                for j in range(len(ys)):
                    keep = max(keep, ys[j])
                    ys[j] = keep

                interf = interp1d(xs, ys)
                max_curve = interf(np.arange(settings['pop_size'] *
                                             settings['generations']))
                curves.append((flop / ((max_curve) * 1e9)))

                gflops.append(np.round(np.max(ys), 2))
                costs.append(np.round(flop / np.max(ys) / 1e9, 6))

            # write to result file
            if not error_occured:
                print("costs: %s\tgflops: %s\telapsed: %.2f" % (
                                    ["%.6f" % x for x in costs],
                                    ["%.2f" % x for x in gflops],
                                    (time.time() - tic) / args.n_ave_curve))

                device_name = tvm.gpu(0).device_name
                n_trial = settings['pop_size'] * settings['generations']
                log_value('cuda', device_name, wkl_name,
                          'TC-%d' % n_trial,
                          array2str_round(curves).replace('inf',
                                                          'float("inf")'))
            else:
                print("error occured. put to retry queue.")
                que.put(wkl)

