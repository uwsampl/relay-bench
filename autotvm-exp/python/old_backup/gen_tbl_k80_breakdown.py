import os
import csv

data = {
    'mobilenet': [],
    'resnet': [],
}

def read_data(fn):
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the title
        for row in reader:
            net = row[0]
            conv = row[1]
            size = int(row[2])
            ci = int(row[3])
            co = int(row[4])
            s = int(row[6])
            op = (conv, size, ci, co, s)
            name = row[7]
            t = float(row[8])

            idx = -1
            for i in range(len(data[net])):
                if data[net][i][0] == op:
                    idx = i
                    break
            if idx >= 0:
                data[net][idx][1][name] = t
            else:
                data[net].append([op, {}])
                data[net][-1][1][name] = t

read_data('../data/K80/tvm_conv2d_perf.csv')
read_data('../data/K80/mxnet_conv2d_perf.csv')

output = 'tbl_k80_resnet.tex'
with open(output, 'w') as f:
    f.write('\\begin{tabular}{lllllccc}\n')
    f.write('\\hline\n')
    f.write('H/W & IC & OC & K & S & \\parbox[r]{.14\\linewidth}{\\centering cuDNN\\\\(ms)} & \\parbox[r]{.15\\linewidth}{\\centering\\TensorOpt\\\\(ms)} & Speedup \\\\ \\hline\n')
    d = data['resnet']
    for op, time in d:
        conv, size, ci, co, s = op
        k = int(conv.split('x')[-1])
        mxnet_time = time['mxnet']
        tvm_time = time['tvm']
        speedup = mxnet_time / tvm_time
        f.write('%s & %s & %s & %s & %s & %.02f & %.02f & %.2f \\\\\n' % (
            size, ci, co, k, s, mxnet_time, tvm_time, speedup))
    f.write('\\hline\n\\end{tabular}\n')
print('output to %s' % output)

output = 'tbl_k80_mobilenet.tex'
with open(output, 'w') as f:
    f.write('\\begin{tabular}{lllllccc}\n')
    f.write('\\hline\n')
    f.write('H/W & C & M & K & S & \\parbox[r]{.14\\linewidth}{\\centering cuDNN\\\\(ms)} & \\parbox[r]{.15\\linewidth}{\\centering\\TensorOpt\\\\(ms)} & Speedup \\\\ \\hline\n')
    d = data['mobilenet']
    for op, time in d:
        conv, size, ci, co, s = op
        if not conv.startswith('dw'):
            continue
        k = int(conv.split('x')[-1])
        mxnet_time = time['mxnet']
        tvm_time = time['tvm']
        speedup = mxnet_time / tvm_time
        f.write('%s & %s & %s & %s & %s & %.02f & %.02f & %.2f \\\\\n' % (
            size, ci, co, k, s, mxnet_time, tvm_time, speedup))
    f.write('\\hline\n\\end{tabular}\n')
print('output to %s' % output)
