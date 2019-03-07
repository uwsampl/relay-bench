import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import matplotlib
matplotlib.rcParams['text.usetex'] = True

from util import query_log_key, query_flop, TVM_NAME

baseline_file = '../baseline/baseline.log'
in_file = '../data/autotuning/tuning_curve.log'

task_names = [
    'resnet.C3.B1'
]

device = 'GeForce GTX TITAN X'

methods = [
    'small_new#xgb-rank',
    'small_new#ga',
    'small_new#random',
    'cudnn',
]

method2color = {
    'small_new#xgb-rank': 'C0',
    'small_new#ga': 'C2',
    'small_new#random': 'C1',
    'cudnn': 'C7',
}

def to_show(name):
    trans_table = {
            'small_new#ga': TVM_NAME + ': Blackbox Genetic Algorithm',
            'small_new#random': TVM_NAME + ': Random Search',
            'small_new#xgb-rank': TVM_NAME + ': ML-based Model',
            'cudnn': 'Baseline: cuDNN',
    }

    return trans_table.get(name, name)


if __name__ == '__main__':
    fontsize = 13
    
    for name in task_names:
        max_trial = 0
        legends = []
        curves = []
        colors = []
        styles = ['solid', 'dashdot', 'dotted', 'dashed']

        baseline = query_log_key('cuda', device, name, 'cudnn', baseline_file)
        baseline = np.mean(eval(baseline))

        fig, ax = plt.subplots()

        for method in methods:
            if 'tvm' in method or '#' in method:
                value = query_log_key('cuda', device, name, method, in_file)
            else:
                value = query_log_key('cuda', device, name, method, baseline_file)

            if value is None:
                print(method)
            costs = np.array(eval(value))
            if len(costs.shape) < 2:
                costs = costs.reshape((len(costs), 1))

            gflops = baseline / costs

            if 'TC' not in method:
                max_trial = min(max(max_trial, len(gflops[0])), 800)

            y = np.mean(gflops, axis=0)

            y_err = np.std(gflops, axis=0)
            if 'cudnn' in method:
                y_err[:] = 0

            curves.append((y, y_err))
            colors.append(method2color[method])
            legends.append(to_show(method))

        for i, curve in enumerate(curves):
            x = np.arange(1, max_trial+1)
            y = np.empty_like(x, dtype=np.float32)
            y_err = np.empty_like(y)

            y_ = curve[0][:len(x)]
            y_err_ = curve[1][:len(x)]

            y[:len(y_)] = y_
            y[len(y_):] = y_[-1]
            y_err[:len(y_err_)] = y_err_
            y_err[len(y_err_):] = y_err_[-1]

            plt.plot(x, y, color=colors[i], linewidth=1.5, linestyle=styles[i])
            plt.fill_between(x, y - y_err, y + y_err, color=colors[i], alpha=0.15)

        plt.legend(legends, fontsize=fontsize - 2)

        ax = plt.gca()
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlim([1, max_trial])
        plt.ylim(ymin=0)
        plt.xlabel("Number of Trials", fontsize=fontsize)
        plt.grid(linewidth=0.4, linestyle='dotted')
        plt.ylabel("Relative Speedup", fontsize=fontsize)

        fig.set_size_inches(5, 3.2)

        out_file = '../figures/tuners.pdf' % ()
        if len(sys.argv) > 1:
            out_file = out_file.replace('.pdf', '_tvm.pdf')

        print("Output to %s" % out_file)
        fig.savefig(out_file, bbox_inches='tight')

        plt.show()

