import sys
sys.path.append("../python")
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
    'cudnn',
    'small_new#random',
    'small_new#xgb-rank',
    #'small_new#ga',
]

method2color = {
    'small_new#xgb-rank': 'C0',
    'small_new#ga': 'C2',
    'small_new#random': 'C1',
    'cudnn': 'C7',
}

method2style  = {
    'small_new#xgb-rank': 'solid',
    'small_new#ga': 'dashdot',
    'small_new#random': 'dotted',
    'cudnn': 'dashed',
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
        styles = []

        baseline = query_log_key('cuda', device, name, 'cudnn', baseline_file)
        baseline = np.mean(eval(baseline))

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
            styles.append(method2style[method])
            legends.append(to_show(method))

        fig, ax = plt.subplots()
        ax = plt.gca()
        plt.xlim([1, max_trial])
        plt.ylim([0, 1.7])
        #plt.xlabel("Number of Trials", fontsize=fontsize)
        plt.grid(linewidth=0.4, linestyle='dotted')
        #plt.ylabel("Relative Speedup", fontsize=fontsize)

        fig.set_size_inches(5, 3.2)


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

            out_file = 'figures/blackbox_vs_ML_%d.pdf' % i
            fig.savefig(out_file, bbox_inches='tight')
            print("Output to %s" % out_file)
            plt.plot(x, y, color=colors[i], linewidth=1.5, linestyle=styles[i])
            plt.fill_between(x, y - y_err, y + y_err, color=colors[i], alpha=0.15)




        out_file = 'figures/blackbox_vs_ML_%d.pdf' % len(curves)
        print("Output to %s" % out_file)
        fig.savefig(out_file, bbox_inches='tight')


