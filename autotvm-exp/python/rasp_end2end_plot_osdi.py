import csv
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['text.usetex'] = True

import numpy as np
from util import query_log_key, query_color, enhance_color, TVM_NAME

baseline_log = '../baseline/baseline.log'
tvm_log = '../data/tvm_result.log'


method2color = {
    'tvm':      enhance_color('C0', l=1.3, s=1),
    'mxnet':    enhance_color('C2', l=1.3, s=1),
    'autotvm0': enhance_color('C9', l=1.3, s=1), 
    'autotvm3': enhance_color('C0', l=1.3, s=1),
    'tflite':   enhance_color('C3', l=1.3, s=1)
}



output = '../figures/rasp_end2end.pdf'
devices = ['Mali-T860']
dtypes = ['float32']
models = ['resnet-18', 'mobilenet', 'nature-dqn'] #'vgg-16']
methods = ['tflite', 'autotvm0', 'autotvm3']

batch_sizes = [1]

def show_name(name):
    trans_table = {
        'autotvm3':    TVM_NAME,
        'autotvm0':    TVM_NAME + ' w/o graph opt',
        'tflite':     'Tensorflow Lite',

        'resnet-18':  'ResNet-18',
        'mobilenet':  'MobileNet',
        'vgg-16':     'VGG-16',
        'nature-dqn': 'DQN',
    }

    return trans_table.get(name, name)

data = {'tflite': {}, 'autotvm0': {}, 'autotvm3': {}}

def read_data(fn):
    with open(fn, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader) # skip the title
        for row in reader:
            net = row[0]
            name = row[1]
            if name == 'tvm' or name == 'mxnet':
                continue
            data[name][net] = float(row[2])

read_data('../data/rasp/end2end_perf.csv')

def query_log_key(target, device, task_name, method, filename='all.log'):
    trans_table = {
            'resnet-18.B1': 'resnet', 'nature-dqn.B1': 'dqn',
            'mobilenet.B1': 'mobilenet',
            }
    task_name = trans_table.get(task_name, task_name)
    return str(data[method][task_name])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    parser.add_argument("--tvm", action='store_true')
    args = parser.parse_args()

    if args.tvm:
        TVM_NAME = 'TVM'
        output = output.replace(".pdf", "_tvm.pdf")

    width = 1
    gap = 1.5
    fontsize = 19

    x0 = 0
    bar_objs = None

    xticks = []
    xlabels = []
    xticks_2 = []
    xlabels_2 = []

    fig, ax = plt.subplots()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    model_to_ax = {
        'resnet-18': ax1,
        'mobilenet': ax1,
        'nature-dqn': ax2,
    }

    for dev in devices:
        for batch_size in batch_sizes:
            for model in models:
                ys = []
                colors = []

                if 'dqn' in model:
                    x0 = 2.2

                baseline = 1e9
                tvm_res = 1e9
                max_height = 0

                for method in methods:
                    if 'B' in model:
                        task_name = model
                    else:
                        task_name = "%s.B%d" % (model, batch_size)
                    in_file = tvm_log if 'tvm' in method else baseline_log 
                    value = query_log_key('opencl', dev, task_name,
                                          method, in_file)
                    if value is None:
                        print("ERROR! cannot find records for",
                              dev, task_name, method)
                        value = "0"
                    value = np.mean(eval(value))

                    if 'tvm' not in method:
                        baseline = min(baseline, value)
                    else:
                        if value != 0:
                            tvm_res = min(tvm_res, value)

                    if 'xla' not in method:
                        max_height = max(max_height, value)

                    ys.append(value)
                    colors.append(method2color[method])

                # draw bar for a model
                xs = np.arange(x0, x0 + len(ys))
                ret = model_to_ax[model].bar(xs, ys, width=width, color=colors)
                if bar_objs is None: # record obj for legend
                    bar_objs = ret
                bar_objs = ret
                x0 += len(ys) * width

                print(model, baseline / tvm_res)
                x0 += gap
                if model_to_ax[model] == ax1:
                    xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
                    xlabels.append(show_name(model))
                else:
                    xticks_2.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
                    xlabels_2.append(show_name(model))

    plt.xticks(xticks)
    for i, t in enumerate(ax1.get_xticklabels()):
        t.set_y(0.00)
        t.set_fontsize(fontsize)

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, fontsize=fontsize)
    ax2.set_xticks(xticks_2)
    ax2.set_xticklabels(xlabels_2, fontsize=fontsize)

    ax1.set_ylim(ymax=800)
    ax1.set_ylabel('Time(ms)', fontsize=fontsize)
    ax2.set_ylim(ymax=12.0)

    for ax in [ax1, ax2]:
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.grid(linewidth=0.4, linestyle='dotted')
        ax.set_axisbelow(True)  # grid lines are behind the rest

    plt.legend(bar_objs, [show_name(x) for x in methods],fontsize=fontsize,
               ncol=3, loc='upper center',
               bbox_to_anchor=(-0.8, 1.3))

    fig.set_size_inches(10, 4)
    fig.tight_layout()

    print("Output to %s" % output)
    fig.savefig(output, bbox_inches='tight')
    if not args.s:
        plt.show()
    plt.close()

