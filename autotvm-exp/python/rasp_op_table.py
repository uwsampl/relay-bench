import argparse

import numpy as np
import csv

# import autotvm.task import workloads  # use this after updating autotvm
resnet_workloads = [
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
]

mobilenet_workloads = [
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

def show_name(dev):
    return dev

tflite_mobilenet_log = '../data/rasp/tflite_depthwise.csv'
tflite_resnet_log = '../data/rasp/tflite_conv.csv'

tvm_mobilenet_log = '../data/rasp/mobilenet_depthwise_rpi3b.csv'
tvm_resnet_log = '../data/rasp/resnet18_rpi3b.csv'

#currently using old style log/csv format:
#layer,shape,dimensions,...,name,time (ms)

devices = ['rpi3b']
models = ['resnet-18', 'mobilenet']
methods = ['tflite', 'tvm']
batch_sizes = [1]

if __name__ == '__main__':
    data = {}

    for dev in devices:
        for model in models:

            workloads = None
            if 'mobilenet' in model:
                workloads = mobilenet_workloads
                tflite_record_path = tflite_mobilenet_log
                tvm_record_path = tvm_mobilenet_log
            elif 'resnet' in model:
                model = 'resnet'
                workloads = resnet_workloads
                tflite_record_path = tflite_resnet_log
                tvm_record_path = tvm_resnet_log
            assert workloads is not None

            if model not in data:
                data[model] = []

            tflite_records = dict()
            tvm_records = dict()
            with open(tflite_record_path, 'r') as tflite_record:
                tflite_csv = csv.reader(tflite_record)
                for row in tflite_csv:
                    key = tuple([int(i) for i in row[0:6]])
                    print(key)
                    assert key not in tflite_records
                    tflite_records[key] = float(row[-1])

            with open(tvm_record_path, 'r') as tvm_record:
                tvm_csv = csv.reader(tvm_record) 
                for row in tvm_csv:
                    print(key)
                    key = tuple([int(i) for i in row[0:6]])
                    assert key not in tvm_records
                    tvm_records[key] = float(row[-1])

            for workload in workloads:
                key = tuple(workload[2:-1])

                costs = dict()
                costs['tflite'] = tflite_records[key]
                costs['tvm'] = tvm_records[key]
    
                data[model].append((workload, costs))

    output = 'resnet_op_%s.tex' % show_name(dev)
    with open(output, 'w') as f:
        f.write('\\begin{tabular}{lcc}\n')
        f.write('\\hline\n')
        f.write('(H/W,IC,OC,K,S) & \\parbox[c]{1.2cm}{\\centering{TFLite\\\\(ms)}} & \\parbox[c]{1.2cm}{\\centering{\\TensorOpt\\\\(ms)}} \\\\ \\hline\n')
        d = data['resnet']
        for wkl, time in d:
            size, ci, co, k, s = wkl[2], wkl[3], wkl[4], wkl[5], wkl[6]
            shape = str((size, ci, co, k, s))
            tflite_time = time['tflite']
            tvm_time = time['tvm']
            speedup = tflite_time / tvm_time
            if speedup > 1:
                tflite_time = '%.02f' % tflite_time
                tvm_time = '%0.2f' % tvm_time
                tvm_time = '\\textbf{' + tvm_time + '}'
            else:
                tflite_time = '%.02f' % tflite_time
                tflite_time = '\\textbf{' + tflite_time + '}'
                tvm_time = '%0.2f' % tvm_time
            f.write('%s & %s & %s \\\\\n' % (
                shape, tflite_time, tvm_time))

        f.write('\\hline\n\\end{tabular}\n')
    print('output to %s' % output)

    output = 'mobilenet_op_%s.tex' % show_name(dev)
    with open(output, 'w') as f:
        f.write('\\begin{tabular}{lcc}\n')
        f.write('\\hline\n')
        f.write('(H/W,C,M,K,S) & \\parbox[c]{1.2cm}{\\centering{TFLite\\\\(ms)}} &  \\parbox[c]{1.2cm}{\\centering{\\TensorOpt\\\\(ms)}} \\\\ \\hline\n')
        d = data['mobilenet']
        for wkl, time in d:
            size, ci, co, k, s = wkl[2], wkl[3], wkl[4], wkl[5], wkl[6]
            shape = str((size, ci, co, k, s))
            tflite_time = time['tflite']
            tvm_time = time['tvm']
            speedup = tflite_time / tvm_time
            if speedup > 1:
                tflite_time = '%.02f' % tflite_time
                tvm_time = '%0.2f' % tvm_time
                tvm_time = '\\textbf{' + tvm_time + '}'
            else:
                tflite_time = '%.02f' % tflite_time
                tflite_time = '\\textbf{' + tflite_time + '}'
                tvm_time = '%0.2f' % tvm_time
            f.write('%s & %s & %s \\\\\n' % (
                shape, tflite_time, tvm_time))
        f.write('\\hline\n\\end{tabular}\n')
    print('output to %s' % output)

