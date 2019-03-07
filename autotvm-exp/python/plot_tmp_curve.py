import argparse
import matplotlib.pyplot as plt

import numpy as np
from util import query_log_key, query_flop, enhance_color, TVM_NAME

import autotvm

in_file = 'to_plot.log'
#device = 'GeForce GTX TITAN X'
device = 'GeForce GTX 1050 Ti'
task_name = 'resnet.C6.B1'
#methods = ['small_new#treernn-reg', 'small_new#treernn-rank', 'small_new#xgb-reg', 'small_new#xgb-rank']
methods = ['transfer-0', 'transfer-1', 'transfer-2']

def show_name(name):
    trans_table = {
        'transfer-0': 'No Transfer',
        'transfer-1': 'Transfer Large',
        'transfer-2': 'Transfer Small',
    }

    return trans_table.get(name, name)

all_curves = []
all_legends = []

x_max = 0
for method in methods:
    value = query_log_key('cuda', device, task_name, method, in_file)
    if value is None:
        continue
    sum_curve = np.array(eval(value))

    flop = query_flop(task_name)
    sum_curve = flop / sum_curve / 1e9

    y = np.mean(sum_curve, axis=0)
    y_err = np.std(sum_curve, axis=0)
    x_max = max(x_max, len(y))

    all_curves.append((y, y_err))
    all_legends.append(show_name(method))

x = np.arange(1, x_max + 1)
for y, y_err in all_curves:
    plt.plot(x, y)
    plt.fill_between(x, y - y_err, y + y_err, alpha=0.15)

plt.legend(all_legends)
ax = plt.gca()
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim([1, x_max])
plt.ylim(ymin=0)
plt.xlabel("n_trial")
plt.ylabel("GFLOPS")
plt.title(task_name)
plt.grid()
plt.show()

