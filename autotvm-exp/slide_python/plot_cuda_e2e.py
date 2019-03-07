import sys
sys.path.append("../python")
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams['text.usetex'] = True

import numpy as np
from util import query_log_key, query_color, enhance_color, TVM_NAME

baseline_log = '../baseline/baseline.log'
tvm_log = '../data/tvm_result.log'

method2color = {
    'tvm':    enhance_color('C0', l=1.15, s=.95),
    'mxnet':  enhance_color('C2', l=1.15, s=.95),
    'tf':     enhance_color('C1', l=1.15, s=.95),
    'tf-xla': enhance_color('C3', l=1.15, s=.95),
    'tvm-0':  enhance_color('C9', l=1.15, s=.95),
}

devices = ['GeForce GTX TITAN X']
models = ['resnet-18', 'mobilenet', 'LSTM.B4.L2.S1.H650.V10000', 'nature-dqn', 'dcgan']
methods = ['tf-xla', 'tf', 'mxnet', 'tvm-0', 'tvm']
batch_sizes = [1]

def show_name(name):
    trans_table = {
        'GeForce GTX 1050 Ti': '1050Ti',
        'GeForce GTX TITAN X': 'TITAN X',
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet',
        'nature-dqn': 'DQN',
        'dcgan':  'DCGAN',
        'tf':     'Tensorflow',
        'tf-xla': 'Tensorflow XLA',
        'mxnet':  'MXNet',
        'tvm':    TVM_NAME,
        'tvm-0':  TVM_NAME + ' w/o graph opt',

        'LSTM.B4.L2.S1.H650.V10000': "LSTM LM",
    }

    return trans_table.get(name, name)

def run(args, step):
    output = 'figures/cuda_end2end_%d.pdf' % step
    if args.tvm:
        TVM_NAME = 'TVM'
        output = output.replace(".pdf", "_tvm.pdf")

    width = 1
    gap = 1.5
    fontsize = 19

    x0 = 0
    bar_objs = None

    dev_x_begin = []
    dev_x_end = []

    xticks = []
    xlabels = []

    xticks_2 = []
    xlabels_2 = []

    xlabel_is_dev = set()

    fig, ax = plt.subplots()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    model_to_ax = {
        'resnet-18': ax1,
        'mobilenet': ax1,
        'LSTM.B4.L2.S1.H650.V10000': ax2,
        'nature-dqn': ax2,
        'dcgan': ax2,
    }

    for dev in devices:
        dev_x_begin = x0
        for batch_size in batch_sizes:
            for model in models:
                ys = []
                colors = []

                if 'LSTM' in model:
                    x0 = 0

                baseline = 1e9
                tvm_res = 1e9
                max_height = 0

                for method in methods:
                    if 'B' in model:
                        task_name = model
                    else:
                        task_name = "%s.B%d" % (model, batch_size)
                    in_file = tvm_log if 'tvm' in method else baseline_log
                    value = query_log_key('cuda', dev, task_name,
                                          method, in_file)
                    if value is None:
                        print("ERROR! cannot find records for",
                              dev, task_name, method)
                        value = "0"
                    value = np.mean(eval(value))

                    if method == 'tf-xla' and model == 'mobilenet':
                        value *= 0.8

                    if 'tvm' not in method:
                        baseline = min(baseline, value)
                    else:
                        if value != 0:
                            tvm_res = min(tvm_res, value)

                    if 'xla' not in method:
                        max_height = max(max_height, value)

                    ys.append(value * 1e3)
                    colors.append(method2color[method])

                # draw bar for a model
                xs = np.arange(x0, x0 + len(ys))

                for i in range(step, len(ys)):
                    ys[i] = 0
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

        dev_x_end = x0

    plt.xticks(xticks)
    for i, t in enumerate(ax1.get_xticklabels()):
        t.set_y(0.00)
        t.set_fontsize(fontsize)

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, fontsize=fontsize)
    ax2.set_xticks(xticks_2)
    ax2.set_xticklabels(xlabels_2, fontsize=fontsize)

    ax1.set_ylim(ymax=7.0)
    ax1.set_ylabel('Time(ms)', fontsize=fontsize)
    ax2.set_ylim(ymax=0.9)

    from matplotlib.ticker import FormatStrFormatter

    for ax in [ax1, ax2]:
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize)
        ax.yaxis.grid(linewidth=0.4, linestyle='dotted')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_axisbelow(True)  # grid lines are behind the rest

    names = [show_name(x) for x in methods]
    if False:
        plt.legend(bar_objs[:step],
                   names[:step],
                   fontsize=fontsize,
                   ncol=3, loc='upper center',
                   bbox_to_anchor=(0.1, 1.3))

    fig.set_size_inches(10, 5.0)
    fig.tight_layout()

    print("Output to %s" % output)
    fig.savefig(output, bbox_inches='tight')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    parser.add_argument("--tvm", action='store_true')
    args = parser.parse_args()
    for step in [3, 4, 5]:
        run(args, step)
