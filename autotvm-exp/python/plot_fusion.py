import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import enhance_color
matplotlib.rcParams['text.usetex'] = True


data = {0: {}, 1: {}}

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
            op = (net, conv, size, ci, co, s)
            opt_lvl = int(row[7])
            t = float(row[8])
            data[opt_lvl][op] = t

read_data('../data/K80/conv2d_bn_relu_perf.csv')
ops = [
    ('resnet', 'conv1x1', 28, 128, 256, 2),
    #('resnet', 'conv3x3', 14, 256, 256, 1),
    #('mobilenet', 'dw_conv3x3', 112, 32, 1, 1),
    ('mobilenet', 'dw_conv3x3', 14, 512, 1, 1),
    ('lm', 'rnn', 128, 0, 0, 0),
    ('lm', 'lstm', 128, 0, 0, 0),
]
op_names = [
    'conv+bn+relu\n128x28x28\n1x1x128x256',
    #'conv2d\n256x14x14\n3x3x256x256',
#    'depthwise-\nconv2d\n32x112x112\n3x3x32',
    'depthwise-\nconv+bn+relu\n512x14x14\n3x3x512',
    'rnn cell\nhidden:128',
    'lstm cell\nhidden:128',
]

width = 0.30
x = np.arange(len(ops)) + 0.2 + width/2
legends = []
labels = ['w/o fusion', 'w/ fusion']
colors = [enhance_color('C2',l=1.3),
          enhance_color('C0',l=1.3)]

fig, ax = plt.subplots()
for opt_lvl in range(2):
    baseline = np.array([data[0][op] for op in ops])
    y = np.array([data[opt_lvl][op] for op in ops])
    y = baseline / y
    b = plt.bar(x, y, width=width, color=colors[opt_lvl], linewidth=0.8, edgecolor='white')
    legends.append(b)
    x += width

plt.xticks(np.arange(len(ops)) + 0.5, op_names, fontsize=10)
plt.tick_params(axis='x', which='both', bottom='off', top='off')
plt.yticks(fontsize=12)
#plt.xlabel('Convolution-BN-Relu Operation', fontsize=12)
plt.ylabel('Relative Speedup', fontsize=12)
ax.yaxis.grid(linewidth=0.4, linestyle='dotted')
ax.set_axisbelow(True)  # grid lines are behind the rest

from matplotlib.ticker import FormatStrFormatter
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


#plt.yscale('log')
plt.legend(legends, labels, fontsize=12)
#plt.show()
output = '../figures/fusion_cmp.pdf'
fig.set_size_inches(5, 2.5)
fig.savefig(output, bbox_inches='tight')
plt.close()
print('output to %s' % output)


