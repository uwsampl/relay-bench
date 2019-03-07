import sys
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
    'tvm':         enhance_color('C0', l=1.2),
    'tvm-float16': enhance_color('C0', l=1.2),
    'ARMComputeLib-gemm-float32': enhance_color('C6', l=1.2),
    'ARMComputeLib-gemm-float16': enhance_color('C6', l=1.2),
    'tvm-0': enhance_color('C9', l=1.2),
    'tvm-0-float16': enhance_color('C9', l=1.2),
}

output = '../figures/mali_end2end.pdf'
device = 'Mali-T860'
dtypes = ['float32', 'float16']
models = ['resnet-18', 'mobilenet', 'nature-dqn'] #'vgg-16']
methods = ['ARMComputeLib-gemm-float32', 'tvm-0', 'tvm']

batch_sizes = [1]

def show_name(name):
    trans_table = {
        'ARMComputeLib-gemm-float32': 'ARMComputeLib',
        'tvm':    TVM_NAME,
        'ARMComputeLib-gemm-float16': 'ARMComputeLib-fp16',
        'tvm-float16':    TVM_NAME + '-fp16',
        'tvm-0': TVM_NAME + ' w/o graph opt',
        'tvm-0-float16': TVM_NAME + '-fp16' + " w/o graph opt",
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet',

        'vgg-16': 'VGG-16',
        'nature-dqn': 'DQN',
    }

    return trans_table.get(name, name)

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

    dev_x_begin = []
    dev_x_end = []

    xticks = []
    xlabels = []
    xlabel_is_dev = set()

    fig, ax = plt.subplots()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 0.9])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    model_to_ax = {
        'resnet-18': ax1,
        'mobilenet': ax1,
        'vgg-16': ax2,
        'nature-dqn': ax2,
    }

    for model in models:
        dev_x_begin = x0
        for batch_size in batch_sizes:
            for dtype in dtypes:
                ys = []
                colors = []
                baseline = 1e9
                tvm_res = 1e9
                max_height = 0
                for method in methods:
                    task_name = "%s.B%d" % (model, batch_size)
                    in_file = tvm_log if 'tvm' in method else baseline_log 

                    if dtype == 'float16':
                        if 'float32' in method:
                            method = method.replace('float32', 'float16')
                        else:
                            method += '-float16'

                    value = query_log_key('opencl', device, task_name, method, in_file)
                    if value is None:
                        print("ERROR! cannot find records for",
                              device, task_name, method)
                        value = '0'
                    value = np.mean(eval(value))

                    if 'tvm' not in method:
                        baseline = min(baseline, value)
                    else:
                        tvm_res = min(tvm_res, value)

                    max_height = max(max_height, value)

                    value *= 1e3
 
                    ys.append(value)
                    colors.append(method2color[method])

                # draw bar for a model
                xs = np.arange(x0, x0 + len(ys))
                ret = model_to_ax[model].bar(xs, ys, width=width, color=colors)
                if bar_objs is None: # record obj for legend
                    bar_objs = ret
                x0 += len(ys) * width
                delta = 0.01
                if 'vgg' in model:
                    delta *= 5
                #model_to_ax[model].annotate( "%.2fx" % (baseline / tvm_res),
                #                            (x0 - width, tvm_res + delta),
                #                            fontsize=fontsize-2,
                #                            ha='center',
                #                            color='black')
                print(model, dtype, baseline/tvm_res)

                x0 += gap

                xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
                xlabels.append(show_name(dtype))
        dev_x_end = x0

        # add device xlabel
        xlabel_is_dev.add(len(xlabels))
        xticks.append(dev_x_begin + (dev_x_end - dev_x_begin) / 2.0 - 1)
        xlabels.append(show_name(model))

    ax1.set_xticks(xticks[:-3])
    ax1.set_xticklabels(xlabels[:-3], fontsize=fontsize)

    ax2.set_xticks(xticks[-3:])
    ax2.set_xticklabels(xlabels[-3:], fontsize=fontsize)

    for i, t in enumerate(ax1.get_xticklabels()):
        if i in xlabel_is_dev:
            t.set_y(-0.08)
            t.set_fontsize(fontsize+1)
        else:
            t.set_y(0.00)
            t.set_fontsize(fontsize)

    for i, t in enumerate(ax2.get_xticklabels()):
        if i in xlabel_is_dev:
            t.set_y(-0.08)
            t.set_fontsize(fontsize+1)
        else:
            t.set_y(0.00)
            t.set_fontsize(fontsize)

    ax1.set_ylim(ymax=250)
    ax1.set_ylabel('Time (ms)', fontsize=fontsize)
    ax2.set_ylim(ymax=5)


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

