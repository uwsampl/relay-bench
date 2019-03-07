import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

data = {'resnet18-fpga-smt': {}, 'resnet18-arm': {}}
blind_review = False

def read_data(fn, key):
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the first row
        for row in reader:
            system = row[0]
            total = float(row[1])
            conv = float(row[2])
            layer_0 = float(row[3])
            other = total-conv-layer_0
            if system in data:
                if key=="total":
                    data[system][key] = total
                elif key=="conv":
                    data[system][key] = conv
                elif key=="layer_0":
                    data[system][key] = layer_0
                elif key=="other":
                    data[system][key] = other

read_data('../data/fpga/vdla_e2e.csv', 'conv')
read_data('../data/fpga/vdla_e2e.csv', 'layer_0')
read_data('../data/fpga/vdla_e2e.csv', 'other')

measurements = ['other', 'layer_0', 'conv']
names = ['resnet18-arm', 'resnet18-fpga-smt']
names_str = {'resnet18-fpga-serial': 'TensorOpt ARM+FPGA (no latency hiding)', 'resnet18-fpga-smt': 'TensorOpt ARM+FPGA', 'resnet18-arm': 'TensorOpt ARM'}
colors = {'other': '#acc2d9', 'layer_0': '#95a3a6', 'conv': '#3c73a8'}

if not blind_review:
    names_str['resnet18-fpga-serial'] = 'TVM ARM+FPGA (no latency hiding)'
    names_str['resnet18-fpga-smt'] = 'TVM ARM+FPGA'
    names_str['resnet18-arm'] = 'TVM ARM'

width = 0.8
gap = 0.2
print(gap)
x = np.arange(len(names)) + gap + width/2
legends = []
labels = [names_str[n] for n in names]

fig, ax = plt.subplots()
bottom = np.zeros(len(names))
for m in measurements:
    y = [data[n][m] for n in names]
    b = plt.bar(x, y, width=width, bottom=bottom, color=colors[m])
    bottom += np.array(y)
    legends.append(b)

plt.ylabel('ResNet18 Inference Time (s)')
plt.xticks(x+width/2, [names_str[n] for n in names], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(legends, measurements, fontsize=12)

# plt.show()
if blind_review:
    output = '../figures/exp_fpga_e2e.pdf'
else:
    output = '../figures/exp_fpga_e2e_tvm.pdf'

fig.set_size_inches(5, 4)
fig.savefig(output, bbox_inches='tight')
plt.close()
print('output to %s' % output)


