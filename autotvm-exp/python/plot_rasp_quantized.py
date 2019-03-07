import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import csv
import colorsys
from util import TVM_NAME

import matplotlib
matplotlib.rcParams['text.usetex'] = True

def enhance_color(color, h=1, l=1, s=1):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))

    h, l, s = h * c[0], l * c[1], s * c[2]
    h, l, s = [max(min(x, 1), 0) for x in [h, l, s]]

    return colorsys.hls_to_rgb(h, l, s)

def plot():
      
    # Data in seconds
    fb_21 = np.genfromtxt('../baseline/rasp/facebook_resnet18_quantized_a2w1.txt', delimiter='\n')
    spatial_21_single = np.genfromtxt('../data/rasp/resnet18_quantized_a2w1_single.csv', delimiter=',')
    spatial_21_parallel = np.genfromtxt('../data/rasp/resnet18_quantized_a2w1.csv', delimiter=',')

    # Colors, Labels
    colors = [enhance_color('y', l=1.2), 
        enhance_color('c', l=1.2), 
        enhance_color('C0', l=1.2)]
    labels = ['Hand optimized', TVM_NAME + ' single-threaded', TVM_NAME + ' multi-threaded']


    # print(spatial_21_single[:, -1])
    # print(fb_21)
    index = np.arange(len(fb_21))
    bar_width = 0.25
    gap = (1 - bar_width * len(labels)) / 2
    fontsize = 19

    fig = plt.figure()
    ax = plt.subplot(111)
    legends = []

    print (spatial_21_single[:, -1] / spatial_21_parallel[:, -1])

    b0 = plt.bar(index, fb_21 / fb_21, width=bar_width, color=colors[0], align='center')
    b1 = plt.bar(index + bar_width, fb_21 / spatial_21_single[:, -1] * 1000, width=bar_width, color=colors[1], align='center')
    b2 = plt.bar(index + 2*bar_width, fb_21 / spatial_21_parallel[:, -1] * 1000, width=bar_width, color=colors[2], align='center')
    legends.append(b0)
    legends.append(b1)
    legends.append(b2)


    ax.set_ylabel('Relative Speedup', fontsize=fontsize)
    xlabels = ('C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12')
    xticks = (index * 1.01 + 0.21)

    plt.xticks(xticks, xlabels, fontsize=fontsize)
    plt.tick_params(axis='x', which='both', bottom='off', top='off')

    plt.yticks(fontsize=fontsize)

    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # grid line
    ax.yaxis.grid(linewidth=0.4, linestyle='dotted')
    ax.set_axisbelow(True)  # grid lines are behind the rest

    plt.legend(legends, labels, fontsize=fontsize-1)

    fig.set_size_inches(10, 4.2)
    output = '../figures/rasp_qnn.pdf'
    if len(sys.argv) > 1:
        output = output.replace('.pdf', '_tvm.pdf')

    fig.savefig(output, bbox_inches='tight')
    #plt.show()
    plt.close()
    print("output to ", output)

plot()
