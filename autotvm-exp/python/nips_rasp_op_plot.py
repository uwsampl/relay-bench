import argparse
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from util import query_log_key, TVM_NAME, enhance_color
import csv
from autotvm.task import name2task

tflite_mobilenet_log = '../data/rasp/tflite_depthwise.csv'
tflite_resnet_log = '../data/rasp/tflite_conv.csv'

tvm_mobilenet_log = '../data/rasp/3mobilenet_depthwise_rpi3b.csv'
tvm_resnet_log = '../data/rasp/3resnet18_rpi3b.csv'

output = '../figures/rasp_op.pdf'

method2color = {
    'tvm':     enhance_color('C0', l=1.3, s=1),
    'tflite':  enhance_color('C3', l=1.3, s=1)
}

device = 'rpi3b'

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

def get_mean(task_name, baseline=True):
    if 'mobilenet' in task_name:
        tflite_record_path = tflite_mobilenet_log
        tvm_record_path = tvm_mobilenet_log
    elif 'resnet' in task_name:
        model = 'resnet'
        tflite_record_path = tflite_resnet_log
        tvm_record_path = tvm_resnet_log


    if baseline:
        record_path = tflite_record_path
    else:
        record_path = tvm_record_path

    records = dict()
    with open(record_path, 'r') as record:
        print(record_path)
        tvm_csv = csv.reader(record) 
        for row in tvm_csv:
            key = tuple([int(i) for i in row[0:6]])
            print(key)
            assert key not in records
            if len(row) == 9:
                records[key] = float(row[-2])
            elif len(row) == 17:
                assert(len(row[7:]) == 10)
                records[key] = np.mean([float(v) for v in row[7:]])
            else:
                assert(False)
   
    task_key = name2task(task_name).args[2:-1]
    print("task_key:", task_key)
    val = records[task_key]
    return val


def show_name(name):
    trans_table = {
        'tvm': TVM_NAME,
        'tflite': 'Tensorflow Lite',
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

        methods = ['tflite', 'tvm']

        x0 = 0
        for task_name in tasks:
            ys = []
            colors = []

            baseline = get_mean(task_name)

            for method in methods:
                value = get_mean(task_name, method=='tflite')

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

    keys = ['tflite', 'tvm']
    # put legend outside the plot
    axes[0].legend([legend_set[x] for x in keys], [show_name(x) for x in keys],
                   fontsize=fontsize,
                   loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4)

    fig.set_size_inches(10, 2.5)

    print("Output to %s" % output)
    fig.savefig(output, bbox_inches='tight')
    if not args.s:
        plt.show()
    plt.close()

