import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True

blind_review = False
names = ['halide', 'tvm no coorp', 'cublas', 'tvm']
names = ['cublas', 'tvm no coorp', 'tvm']
names_str = {
    #'halide': 'Halide',
    'tvm': 'TensorOpt',
    'tvm no coorp': 'TensorOpt w/o coop.',
    'cublas': 'cuBLAS'
}

if not blind_review:
    names_str = {
        'halide': 'Halide',
        'tvm': 'TVM',
        'tvm no coorp': 'TVM w/o coop.',
        'cublas': 'cuBLAS'
        }

colors = {
    'halide': 'C3',
    'tvm': 'C0',
    'tvm no coorp': 'C1',
    'cublas': 'C2'
}
data = {
    'halide': {},
    'tvm no coorp': {},
    'tvm': {},
    'cublas': {}
}

device = 'K80'

def read_data(fn):
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the title
        for row in reader:
            name = row[0]
            size = int(row[1])
            t = float(row[2])
            #gflops = float(row[2])
            data[name][size] = t

read_data('../data/TitanX/gemm_tvm.csv')

fig, ax = plt.subplots()
sizes = sorted(data['tvm'].keys())
width = 0.2
x = np.arange(len(sizes)) + 0.1 + width/2
legends = []
labels = [names_str[n] for n in names]

for name in names:
    y = [data[name][s] for s in sizes]
    b = plt.bar(x, y, width=width, color=colors[name])
    legends.append(b)
    x += width

plt.xticks(np.arange(len(sizes)) + 0.5, sizes, fontsize=12)
plt.tick_params(axis='x', which='both', bottom='off', top='off')
plt.yticks(fontsize=12)
plt.xlabel('Matrix Size', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
plt.legend(legends, labels, fontsize=11)
if not blind_review:
    output = '../figures/gemm_cmp_tvm.pdf'
else:
    output = '../figures/gemm_cmp.pdf'

fig.set_size_inches(5, 3.5)
fig.savefig(output, bbox_inches='tight')
plt.close()
print('output to %s' % output)
