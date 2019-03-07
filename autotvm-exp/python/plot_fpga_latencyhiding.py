import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = {'resnet[1]': {},'resnet[2]': {},'resnet[3]': {},'resnet[4]': {},'resnet[5]': {},'resnet[6]': {},'resnet[7]': {},'resnet[8]': {},'resnet[9]': {},'resnet[10]': {},'resnet[11]': {}}

def read_data(fn, lat_hiding):
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the first row
        for row in reader:
            layer = row[0]
            lat_hid_str = row[1]
            time = float(row[2])
            if layer in data and lat_hiding==eval(lat_hid_str):
                data[layer][lat_hid_str] = time

read_data('../data/fpga/fpga_conv2d_cost.csv', True)
read_data('../data/fpga/fpga_conv2d_cost.csv', False)

layers = ['resnet[1]' ,'resnet[2]' ,'resnet[3]' ,'resnet[4]' ,'resnet[5]' ,'resnet[6]' ,'resnet[7]' ,'resnet[8]' ,'resnet[9]' ,'resnet[10]' ,'resnet[11]' ]
names = ['False', 'True']
names_str = {'False': 'Baseline', 'True': 'Latency Hiding'}
colors = {'False': '#acc2d9', 'True': '#3c73a8'}

width = 0.2
gap = (1 - width * len(names)) / 2
x = np.arange(len(layers)) + gap + width/2
legends = []
labels = [names_str[n] for n in names]

fig, ax = plt.subplots()
for n in names:
    x += width
    y = [data[l][n] for l in layers]
    b = plt.bar(x, y, width=width, color=colors[n])
    legends.append(b)

plt.ylabel('Inference Time (ms)')
plt.xticks(x, layers, fontsize=12, rotation=70)
plt.yticks(fontsize=12)
plt.legend(legends, [names_str[n] for n in names], fontsize=12, loc='best')

# plt.show()
output = '../figures/exp_fpga_latencyhiding.pdf'

fig.set_size_inches(5, 2)
fig.savefig(output, bbox_inches='tight')
plt.close()
print('output to %s' % output)


