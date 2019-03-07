import argparse

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np
from util import query_log_key, query_flop, enhance_color, TVM_NAME

tvm_result_file = '../data/nips_curves.log'
baseline_file = '../baseline/baseline.log'
output = '../figures/cost_model.pdf'

task_names = [
    'resnet.C7.B1',
    'resnet.C8.B1',
    'resnet.C7.B1',
    'resnet.C7.B1',
]

methods = [
    'cudnn',
    'ARMComputeLib-float32',
    'tf-lite',

    # cuda
    'TC-5000',
    'small_new-w#random',
    #'small_new-w#ga',
    'small_new-w#xgb-rank',
    'small_new#xgb-rank-tl',

    # mali
    'spatial_pack#random',
    #'spatial_pack#ga',
    'spatial_pack#xgb-rank',
    'spatial_pack#xgb-rank-tl',

    # rasp
]

method2color = {
    # cuda
    'small_new#xgb-rank-tl': 'C1',
    'small_new-w#xgb-rank':    'C0',
    'small_new-w#random':      'C2',
    'small_new-w#ga':          'C3',
    'cudnn':                 'C7',
    'TC-5000':               'C5',

    'spatial_pack#xgb-rank':    'C0',
    'spatial_pack#xgb-rank-tl': 'C1',
    'spatial_pack#random':      'C2',
    'spatial_pack#ga':          'C3',
    'ARMComputeLib-float32':    'C7',
    'tf-lite':                  'C7',
}

TC_SECOND_PER_TRIAL = 1.3

def show_name(name):
    trans_table = {
        'small_new#xgb-rank-tl': TVM_NAME + " Transfer",
        'small_new-w#xgb-rank': TVM_NAME,
        'small_new-w#random':   'Random',
        'small_new-w#ga':       'GA',

        'spatial_pack#xgb-rank': TVM_NAME,
        'spatial_pack#xgb-rank-tl': TVM_NAME + ' Transfer',
        'spatial_pack#random':   'Random',
        'spatial_pack#ga':       'GA',

        'ARMComputeLib-float32': 'Baseline',
        'cudnn': 'Baseline',
        'tf-lite': 'Baseline',
        'TC-5000': 'TensorComprehensions',

        'GeForce GTX TITAN X': "NVIDIA TITAN X",
        'Mali-T860': "ARM Mali-T860",
        'rpi3b': "ARM Cortex-A53",
    }

    return trans_table.get(name, name)

CUDNN_CORRECT_K = 1.14  # to correct the scaling effect of min_repeat_ms

def get_baseline(target, device, task_name):
    flop = query_flop(task_name)

    if target == 'cuda':
        method = 'cudnn'
    elif target == 'opencl':
        method = 'ARMComputeLib-float32'
    elif target == 'llvm':
        method = 'tf-lite'

    value = query_log_key(target, device, task_name, method, baseline_file)
    value = np.mean(eval(value))

    if method == 'cudnn':
        value *= CUDNN_CORRECT_K

    return flop / value / 1e12

def get_device(k):
    if k in [0, 1]:
        return 'cuda', 'GeForce GTX TITAN X'
    elif k == 2:
        return 'llvm', 'rpi3b'
    elif k == 3:
        return 'opencl', 'Mali-T860'

def get_T_MAX(k):
    if k in [0, 1]:
        return 1000
    elif k == 2:
        return 300
    elif k == 3:
        return 300

def draw(task_names, methods, output, show_name, args, x_max=None):
    COL = 4

    n_tasks = len(task_names)
    fig, axes = plt.subplots((n_tasks + COL-1) // COL, COL)
    axes = axes.flatten()

    fontsize = 10
    legend_ax = min(1, n_tasks - 1)
    objs = []
    all_legends = []

    for k, task_name in enumerate(task_names):
        all_curves = []
        all_colors = []
        ax = axes[k]

        T_MAX = get_T_MAX(k)
        x_time = np.arange(T_MAX)

        for method in methods:
            raw_method = method
            if '*' in method:
                method, step = method.split('*')
                step = int(step)
            else:
                step = 1

            if 'tvm' in method or '#' in method:
                in_file = tvm_result_file
            else:
                in_file = baseline_file

            target, device = get_device(k)
            value = query_log_key(target, device, task_name, method, in_file)

            if value is None:
                print("cannot find", task_name, method)
                continue

            flop = query_flop(task_name)

            if 'cudnn' in method or 'ARMComputeLib' in method or 'tf-lite' in method:
                value = np.array(eval(value))
                value = flop / value / 1e12

                baseline = get_baseline(target, device, task_name)
                if 'cudnn' in method:
                    baseline *= CUDNN_CORRECT_K
                value /= baseline
                y = np.mean(value) * np.ones_like(x_time)
                y_err = np.std(value) * np.ones_like(x_time)
            else:
                cost_curve = eval(value)

                if 'TC' in method:
                    value = eval(query_log_key(target, device, task_name, method + '-time', in_file))
                    per_sample = value / np.array(cost_curve).size
                    stamp_curve = [np.arange(len(cost_curve[0])) * per_sample for _ in range(len(cost_curve))]
                else:
                    value = query_log_key(target, device, task_name, method + '-time', in_file)
                    stamp_curve = eval(value)

                flops_curve = []
                for i in range(len(cost_curve)):
                    # fix time error caused by replay
                    def fix_max(aha):
                        keep = 0
                        for j in range(len(aha)):
                            keep = max(keep, aha[j])
                            aha[j] = keep

                    t = np.array(stamp_curve[i])
                    t -= t[0]
                    fix_max(t)

                    # if target in ['cuda'] and 'tl' in method:
                    #     t = np.arange(len(t)) * 1000.0 / (len(t))

                    y = np.array(cost_curve[i])
                    y = flop / y / 1e12
                    fix_max(y)

                    t = np.concatenate(([0], t))
                    y = np.concatenate(([0], y))
                    if np.max(t) < np.max(x_time):
                        print("aha", np.max(t), np.max(x_time))
                        t = np.concatenate((t, [np.max(x_time)+1]))
                        y = np.concatenate((y, [y[-1]]))
                    interp_f = interp1d(t, y, fill_value=-1)
                    flops_curve.append(interp_f(x_time * step))
                    fix_max(flops_curve[-1])

                flops_curve = np.array(flops_curve)

                baseline = get_baseline(target, device, task_name)
                flops_curve /= baseline

                y = np.mean(flops_curve, axis=0)
                y_err = np.std(flops_curve, axis=0)

            all_curves.append((y, y_err))

            if k == legend_ax:
                all_legends.append(show_name(raw_method))
            all_colors.append(method2color[raw_method])

        for i, (y, y_err) in enumerate(all_curves):
            obj = ax.plot(x_time, y, color=all_colors[i])
            #ax.fill_between(x_time, y - y_err, y + y_err, alpha=0.15)

            if k == legend_ax:
                objs.append(obj[0])

        ax.set_title(task_name.split('.')[1] + ' on ' + show_name(device))
        ax.set_xlim([0, T_MAX])
        if k // COL == (len(task_names)-1) // COL:
            ax.set_xlabel("Time (second)")
        if k % COL == 0:
            ax.set_ylabel("Relative Speedup")

    def swap(a, i, j):
        a[i], a[j] = a[j], a[i]

    fig.set_size_inches(10, 2.5 * len(task_names) // COL)

    if args.full:
        offset = (1.2, 1.35)
        axes[legend_ax].legend(objs, all_legends,
                               loc='upper center', bbox_to_anchor=offset,
                               ncol=len(objs))
    else:
        offset = (1.2, 1.55)
        axes[legend_ax].legend(objs, all_legends,
                               ncol=len(objs),
                               loc='upper center', bbox_to_anchor=offset)


    fig.set_size_inches(10, 2.5 * len(task_names) // COL)
    fig.tight_layout()
    fig.savefig(output, bbox_inches='tight')

    if not args.s:
        plt.show()
    plt.close()


    print("Output to %s" % output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument("--s", action='store_true')
    parser.add_argument("--tvm", action='store_true')
    args = parser.parse_args()

    if args.full:
        output = '../figures/wall_clock.pdf'
    else:
        output = '../figures/wall_clock.pdf'

    if args.tvm:
        TVM_NAME = 'AutoTVM'
        output = output.replace(".pdf", "_tvm.pdf")

    draw(task_names, methods, output, show_name, args)

