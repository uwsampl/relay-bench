import argparse
import matplotlib.pyplot as plt

import numpy as np
from util import query_log_key, array2str_round, query_flop

# import autotvm.task import workloads  # use this after updating autotvm
workloads = [
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

    ("mobilenet.D1.B1", 1, 112, 32, 1, 3, 1, 1, "float32"),
    ("mobilenet.D2.B1", 1, 112, 64, 1, 3, 2, 1, "float32"),
    ("mobilenet.D3.B1", 1, 56, 128, 1, 3, 1, 1, "float32"),
    ("mobilenet.D4.B1", 1, 56, 128, 1, 3, 2, 1, "float32"),
    ("mobilenet.D5.B1", 1, 28, 256, 1, 3, 1, 1, "float32"),
    ("mobilenet.D6.B1", 1, 28, 256, 1, 3, 2, 1, "float32"),
    ("mobilenet.D7.B1", 1, 14, 512, 1, 3, 1, 1, "float32"),
    ("mobilenet.D8.B1", 1, 14, 512, 1, 3, 2, 1, "float32"),
    ("mobilenet.D9.B1", 1, 7, 1024, 1, 3, 1, 1, "float32"),
]

def show_name(name):
    trans_table = {}
    if name in trans_table:
        return trans_table[name]
    return name

task_names = [
    "mobilenet.D1.B1", "mobilenet.D2.B1",
    "mobilenet.D3.B1", "mobilenet.D4.B1",
    "mobilenet.D5.B1", "mobilenet.D6.B1",
    "mobilenet.D7.B1", "mobilenet.D8.B1",
    "mobilenet.D9.B1",

    "resnet.C1.B1", "resnet.C2.B1",
    "resnet.C3.B1", "resnet.C4.B1",
    "resnet.C5.B1", "resnet.C6.B1",
    "resnet.C7.B1", "resnet.C8.B1",
    "resnet.C9.B1", "resnet.C10.B1",
    "resnet.C11.B1", "resnet.C12.B1",
]

baseline_log = '../baseline/baseline.log'
tvm_log = '../data/tvm_result.log'

devices = ['Mali-T860']
models = ['resnet-18', 'mobilenet']
methods = ['ARMComputeLib-float32', 'tvm']
batch_sizes = [1]

def get_op_info(name):
    for wkl in workloads:
        if wkl[0] == name:
            return wkl

if __name__ == '__main__':
    data = {}

    for dev in devices:
        for name in task_names:
            if '.C' in name:
                model = 'resnet'
            elif '.D' in name:
                model = 'mobilenet'

            costs = {}
            for method in methods:
                in_file = tvm_log if 'tvm' in method else baseline_log 
                value = query_log_key('opencl', dev, name,
                                      method, in_file)
                if value is None:
                    print("ERROR! cannot find records for",
                          dev, name, method)
                    continue

                model = name.split('.')[0]
                if model not in data:
                    data[model] = []
                cost = np.min(eval(value)) * 1e3

                costs[method] = cost
            data[model].append((get_op_info(name), costs))

    output = 'resnet_op_%s.tex' % show_name(dev)
    with open(output, 'w') as f:
        f.write('\\begin{tabular}{lllllccc}\n')
        f.write('\\hline\n')
        f.write('H/W & IC & OC & K & S & ARMComputeLib & TensorOpt & Speedup \\\\ \\hline\n')
        d = data['resnet']
        for wkl, time in d:
            size, ci, co, k, s = wkl[2], wkl[3], wkl[4], wkl[5], wkl[6]
            acl_time = time['ARMComputeLib-float32']
            tvm_time = time['tvm']
            speedup = acl_time / tvm_time
            f.write('%s & %s & %s & %s & %s & %.02f & %0.2f & %.2f \\\\\n' % (
                size, ci, co, k, s, acl_time, tvm_time, speedup))
        f.write('\\hline\n\\end{tabular}\n')
    print('output to %s' % output)

    output = 'mobilenet_op_%s.tex' % show_name(dev)
    with open(output, 'w') as f:
        f.write('\\begin{tabular}{lllllccc}\n')
        f.write('\\hline\n')
        f.write('H/W & C & M & K & S & ARMComputeLib & TensorOpt & Speedup \\\\ \\hline\n')
        d = data['mobilenet']
        for wkl, time in d:
            size, ci, co, k, s = wkl[2], wkl[3], wkl[4], wkl[5], wkl[6]
            acl_time = time['ARMComputeLib-float32']
            tvm_time = time['tvm']
            speedup = acl_time / tvm_time
            f.write('%s & %s & %s & %s & %s & %.02f & %.02f & %.2f \\\\\n' % (
                size, ci, co, k, s, acl_time, tvm_time, speedup))
        f.write('\\hline\n\\end{tabular}\n')
    print('output to %s' % output)

