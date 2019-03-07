import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

import numpy as np
from util import query_log_key, query_color, enhance_color, TVM_NAME

baseline_log = '../baseline/baseline.log'
tvm_log = '../data/tvm_result.log'
output = '../figures/cuda_op.pdf'

method2color = {
    'tvm':      enhance_color('C0', l=1.3, s=1.1),
    'TC-1000':  enhance_color('C5', l=1.3, s=1.1),
    'TC-5000':  enhance_color('C5', l=1.3, s=1.1),
    'cudnn':    enhance_color('C7', l=1.3, s=1.1),
    'mxnet':    enhance_color('C2', l=1.3, s=1.1),
    'tvm-winograd': enhance_color('C9', l=1.1, s=1.1),
}

device = 'GeForce GTX TITAN X'

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


def show_name(name):
    trans_table = {
        'GeForce GTX TITAN X': 'TITAN X',
        'cudnn':  'cuDNN',
        'TC-1000': 'TensorComprehensions',
        'TC-5000': 'TensorComprehensions',
        'tvm': TVM_NAME,
        'tvm-winograd': TVM_NAME + " PT",
        'mxnet': 'MX Kernel',
    }

    if 'resnet' in name or 'mobilenet' in name:
        return name.split(".")[1]

    return trans_table.get(name, name)

if __name__ == '__main__':
    width = 1
    gap = 1.5
    fontsize = 19

    fig, ax = plt.subplots()
    legend_set = {}
    
    axes = []

    for k, tasks in enumerate([conv_tasks, dw_conv_tasks]):
        ax = plt.subplot(2, 1, k+1)
        axes.append(ax)

        xticks = []
        xlabels = []

        if tasks == conv_tasks:
            methods = ['cudnn', 'TC-5000', 'tvm', 'tvm-winograd']
        else:
            methods = ['mxnet', 'TC-1000', 'tvm']

        x0 = 0
        for task_name in tasks:
            ys = []
            colors = []

            # read baseline
            if 'C' in task_name:
                baseline = query_log_key('cuda', device, task_name, 'cudnn', baseline_log)
            elif 'D' in task_name:
                baseline = query_log_key('cuda', device, task_name, 'mxnet', baseline_log)
            baseline = np.mean(eval(baseline))

            for method in methods:
                in_file = tvm_log if 'tvm' in method else baseline_log 
                value = query_log_key('cuda', device, task_name, method, in_file)
                if value is None:
                    print("ERROR! cannot find records for", device, task_name, method)
                    continue

                if 'TC' in method: # TC is tuning curve, so we pick the minimul time cost
                    value = np.min(eval(value))
                else: # Others are direct results
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

  
    keys = ['cudnn', 'TC-5000', 'mxnet', 'tvm', 'tvm-winograd']
    # put legend outside the plot
    axes[0].legend([legend_set[x] for x in keys], [show_name(x) for x in keys],
                   fontsize=fontsize-1,
                   loc='upper center', bbox_to_anchor=(0.5, 1.70), ncol=2)

    if len(sys.argv) > 1:
        output = output.replace('.pdf', '_tvm.pdf')

    fig.set_size_inches(10, 6)
    fig.savefig(output, bbox_inches='tight')
    #plt.show()
    print("Output to %s" % output)

