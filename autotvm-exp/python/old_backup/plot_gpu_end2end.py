import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = {'tvm': {}, 'mxnet': {}, 'tf': {}, 'tf-xla': {}}
blind_review = False

def read_data(fn, name, gpu):
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the title
        for row in reader:
            if name == 'tvm':
                net = row[0]
                if net == 'resnet-18':
                    net = 'resnet'
                opt = int(row[1])
                t = float(row[2])
                key = '%s/%s' % (net, gpu)
                if opt == 2:
                    data[name][key] = t
            elif name == 'tf':
                net = row[0]
                xla = eval(row[1])
                t = float(row[2])
                key = '%s/%s' % (net, gpu)
                if xla:
                    data[name + '-xla'][key] = t
                else:
                    data[name][key] = t
            else:
                net = row[0]
                if net == 'resnet-18':
                    net = 'resnet'
                key = '%s/%s' % (net, gpu)
                t = float(row[1])
                data[name][key] = t

read_data('../data/K80/tvm_end2end_perf.csv', 'tvm', 'k80')
read_data('../data/K80/tf_end2end_perf.csv', 'tf', 'k80')
read_data('../data/K80/mxnet_end2end_perf.csv', 'mxnet', 'k80')
read_data('../data/GTX1080/tvm_end2end_perf.csv', 'tvm', '1080')
read_data('../data/GTX1080/tf_end2end_perf.csv', 'tf', '1080')
read_data('../data/GTX1080/mxnet_end2end_perf.csv', 'mxnet', '1080')

nets = ['resnet/k80', 'mobilenet/k80', 'resnet/1080', 'mobilenet/1080']
names = ['tf-xla', 'tf', 'mxnet', 'tvm']
names_str = {'tvm': 'TensorOpt', 'mxnet': 'MXNet', 'tf': 'Tensorflow', 'tf-xla': 'Tensorflow XLA'}
colors = {'tvm': 'C0', 'mxnet': 'C1', 'tf': 'C2', 'tf-xla': 'C3'}

if not blind_review:
    names_str['tvm'] = 'TVM'

width = 0.2
gap = (1 - width * len(names)) / 2
x = np.arange(len(nets)) + gap + width/2
legends = []
labels = [names_str[n] for n in names]

fig, ax = plt.subplots()
for name in names:
    y = [data[name][n] for n in nets]
    b = plt.bar(x, y, width=width, color=colors[name])
    legends.append(b)
    x += width

xlbls = ['ResNet', 'K80', 'MobileNet', 'ResNet', 'GTX1080', 'MobileNet']
xticks = [0.5, 1, 1.5, 2.5, 3, 3.5]
#plt.xticks(np.arange(len(nets)) + 0.5, nets_str, fontsize=12)
plt.xticks(xticks, xlbls, fontsize=12)
va = [0, -0.08, 0, 0, -0.08, 0]
fontsizes = [11, 12, 11, 11, 12, 11]
for t, y, fs in zip(ax.get_xticklabels( ), va, fontsizes):
    t.set_y(y)
    t.set_fontsize(fs)
plt.tick_params(axis='x', which='both', bottom='off', top='off')
plt.yticks(fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
#plt.yscale('log')
plt.legend(legends, labels, fontsize=12)
#plt.show()
if blind_review:
    output = '../figures/gpu_end2end_cmp.pdf'
else:
    output = '../figures/gpu_end2end_cmp_tvm.pdf'

fig.set_size_inches(5, 4)
fig.savefig(output, bbox_inches='tight')
plt.close()
print('output to %s' % output)


