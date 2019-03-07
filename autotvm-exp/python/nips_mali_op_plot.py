import argparse
import sys
import matplotlib.pyplot as plt

import numpy as np
from util import query_log_key, query_color, enhance_color, TVM_NAME

from mali_end2end_plot import method2color

baseline_log = '../baseline/baseline.log'
tvm_log = '../data/tvm_result.log'
output = '../figures/mali_op.pdf'

device = 'Mali-T860'

conv_tasks = [
    'resnet.C1.B1', 'resnet.C2.B1', 'resnet.C3.B1',
    'resnet.C4.B1', 'resnet.C5.B1', 'resnet.C6.B1',
    'resnet.C7.B1', 'resnet.C8.B1', 'resnet.C9.B1',
    'resnet.C10.B1', 'resnet.C11.B1', 'resnet.C12.B1',
]
dw_conv_tasks = [
    'mobilenet.D1.B1', 'mobilenet.D2.B1', 'mobilenet.D3.B1',
    'mobilenet.D4.B1', 'mobilenet.D5.B1', 'mobilenet.D6.B1',
    'mobilenet.D7.B1', 'mobilenet.D8.B1', 'mobilenet.D9.B1',
]

method2color['ARMComputeLib-float32'] = method2color['ARMComputeLib-gemm-float32']

def show_name(name):
    trans_table = {
        'ARMComputeLib-gemm-float32': 'ARMComputeLib',
        'ARMComputeLib-float32': 'ARMComputeLib',
        'tvm':    TVM_NAME,
        'ARMComputeLib-gemm-float16': 'ARMComputeLib-fp16',
        'tvm-float16':    TVM_NAME + '-fp16',
        'tvm-0': TVM_NAME + ' w/o graph opt',
        'tvm-0-float16': TVM_NAME + '-fp16' + " w/o graph opt",
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet',
    }

    if 'resnet' in name or 'mobilenet' in name:
        return name.split(".")[1]

    return trans_table.get(name, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    parser.add_argument("--tvm", action='store_true')
    args = parser.parse_args()

    if args.tvm:
        TVM_NAME = 'AutoTVM'
        output = output.replace(".pdf", "_tvm.pdf")

    width = 1
    gap = 1.5
    fontsize = 19

    fig, ax = plt.subplots()
    legend_set = {}
    
    axes = []

    for k, tasks in enumerate([conv_tasks]):
        ax = plt.subplot(1, 1, k+1)
        axes.append(ax)

        xticks = []
        xlabels = []

        methods = ['ARMComputeLib-float32', 'tvm']

        x0 = 0
        for task_name in tasks:
            ys = []
            colors = []

            # read baseline
            baseline = query_log_key('opencl', device, task_name,
                                     'ARMComputeLib-float32', baseline_log)
            baseline = np.mean(eval(baseline))

            for method in methods:
                in_file = tvm_log if 'tvm' in method else baseline_log 
                value = query_log_key('opencl', device, task_name, method, in_file)
                if value is None:
                    print("ERROR! cannot find records for", device, task_name, method)
                    continue

                value = np.mean(eval(value))

                ys.append(baseline / value)
                colors.append(method2color[method])

            # draw bars for a task
            xs = np.arange(x0, x0 + len(ys))
            bars = plt.bar(xs, ys, width=width, color=colors)
            for method, bar_obj in zip(methods, bars):
                if method not in legend_set:
                    legend_set[method] = bar_obj
            x0 += len(ys)
            x0 += gap

            xticks.append(x0 - gap - len(ys)*width/2.0 - width/2.0)
            xlabels.append(show_name(task_name))

        # tick and label
        plt.xticks(xticks, xlabels, fontsize=fontsize)
        plt.tick_params(axis='x', which='both', bottom='off', top='off')

        plt.ylabel('Relative Speedup', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # grid line
        ax.yaxis.grid(linewidth=0.4, linestyle='dotted')
        ax.set_axisbelow(True)  # grid lines are behind the rest

  
    keys = ['ARMComputeLib-float32', 'tvm']
    # put legend outside the plot
    axes[0].legend([legend_set[x] for x in keys], [show_name(x) for x in keys],
                   fontsize=fontsize-1,
                   loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2)

    fig.set_size_inches(10, 2.5)

    print("Output to %s" % output)
    fig.savefig(output, bbox_inches='tight')
    if not args.s:
        plt.show()
    plt.close()

