import time
import numpy as np
from itertools import product
import csv


def write_row(writer, fieldnames, fields):
    record = {}
    for i in range(len(fieldnames)):
        record[fieldnames[i]] = fields[i]
    writer.writerow(record)


def score_loop(num, trial, trial_args, setup_args, n_times, dry_run, writer, fieldnames):
    for i in range(dry_run + n_times):
        if i == dry_run:
            tic = time.time()
        start = time.time()
        out = trial(*trial_args)
        end = time.time()
        length = end - start
        if i >= dry_run:
            write_row(writer, fieldnames, setup_args + [num, i - dry_run, length])
    final = time.time()

    return (final - tic) / n_times


def run_trials(method, task_name,
               dry_run, times_per_input, n_input,
               trial, trial_setup, trial_teardown,
               parameter_names, parameter_ranges):
    filename = '{}-{}.csv'.format(method, task_name)
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = parameter_names + ['run', 'rep', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for args in product(*parameter_ranges):
            while True:
                costs = []
                for t in range(n_input):
                    trial_args = trial_setup(*args)
                    score = score_loop(t, trial, trial_args, list(args), times_per_input, dry_run, writer, fieldnames)
                    trial_teardown(*trial_args)

                    if t != n_input - 1:
                        time.sleep(4)
                    costs.append(score)

                if np.std(costs) / np.mean(costs) < 0.04:
                    break
                print(costs, 'retry due to high variance in measure results')

            log_value(method, task_name, '',  parameter_names, args, array2str_round(costs))
            print(method, task_name, args, ["%.6f" % x for x in costs])


def run_experiments(experiment, n_ave_curve,
                    method, task_name, device_name,
                    parameter_names,
                    parameter_ranges):
    for args in product(*parameter_ranges):
        while True:
            costs = []
            for t in range(n_ave_curve):
                score = experiment(*args)

                if t != n_ave_curve - 1:
                    time.sleep(4)
                costs.append(1 / score)

            if np.std(costs) / np.mean(costs) < 0.04:
                break
            print(costs, 'retry due to high variance in measure results')

        log_value(method, task_name, device_name, parameter_names, args, array2str_round(costs))
        print(method, task_name, args, ["%.6f" % x for x in costs])


def array2str_round(x, decimal=6):
    """ print an array of float number to pretty string with round

    Parameters
    ----------
    x: Array of float or float
    decimal: int
    """
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return "[" + ", ".join([array2str_round(y, decimal=decimal)
                                for y in x]) + "]"
    format_str = "%%.%df" % decimal
    return format_str % x


def log_value(device_name, task_name, method, parameter_names, parameter_values,
              value, out_file='tmp.log'):
    log_line = "\t".join([str(x) for x in [method, task_name, device_name,
                                           parameter_names, parameter_values,
                                           value, time.time()]])

    with open(out_file, 'a') as fout:
        fout.write(log_line + "\n")

# def log_value(device, backend, task_type, workload, method, template, value, out_file='tmp.log'):
#     """
#     append experiment result to a central log file

#     Parameters
#     ----------
#     device: str
#     backend: str
#     task_type: str
#     workload: str
#     method: str
#     template: str
#     value: str
#     out_file: str
#     """
#     log_line = "\t".join([str(x) for x in [device, backend, task_type, workload, method, template, value, time.time()]])
#     with open(out_file, 'a') as fout:
#         fout.write(log_line + "\n")


def query_log_key(target, device, task_name, method, filename='all.log'):
    """ query value from uniform experiment log file
    the records in file should be logged by autotvm-exp/util/util.py log_value

    Parameters
    ----------
    target: str
        one of 'cuda', 'opencl', 'llvm'
    device: str
        return string by TVMContext.device_name
    task_name: str
    method: str
    filename: str
    """
    finds = []
    wanted = ''.join((target, device, task_name, method))
    with open(filename) as fin:
        for line in fin.readlines():
            items = line.split('\t')
            if len(items) != 6:
                continue
            target, device, task_name, method, value, tstamp = items
            key = ''.join((target, device, task_name, method))

            if key == wanted:
                finds.append(value)

    if finds:
        return finds[-1]
    else:
        return None


def query_flop(task_name):
    """
    Query number of float operation of a task.
    use this function to avoid the dependency of autotvm

    Parameters
    ----------
    task_name: string

    Returns
    ------
    flop: int
    """
    res_table = {
        "resnet.C1.B1": 236027904,
        "resnet.C2.B1": 231211008,
        "resnet.C3.B1": 25690112,
        "resnet.C4.B1": 115605504,
        "resnet.C5.B1": 12845056,
        "resnet.C6.B1": 231211008,
        "resnet.C7.B1": 115605504,
        "resnet.C8.B1": 12845056,
        "resnet.C9.B1": 231211008,
        "resnet.C10.B1": 115605504,
        "resnet.C11.B1": 12845056,
        "resnet.C12.B1": 231211008,

        'mobilenet.D1.B1': 7225344,
        'mobilenet.D2.B1': 3612672,
        'mobilenet.D3.B1': 7225344,
        'mobilenet.D4.B1': 1806336,
        'mobilenet.D5.B1': 3612672,
        'mobilenet.D6.B1': 903168,
        'mobilenet.D7.B1': 1806336,
        'mobilenet.D8.B1': 451584,
        'mobilenet.D9.B1': 903168,

        "other.DEN1": 1024 * 1024 * 1024 * 2,
    }

    if task_name.count('.') == 3:
        task_name = task_name[:task_name.rindex('.')]

    return res_table[task_name]

def query_color(color):
    trans_table = {
        'blue':   '#7cb5ec',
        'black':  '#434348',
        'green':  '#90ed7d',
        'orange': '#f7a35c',
        'purple': '#8085e9',
        'brown':  '#8d6e63',
        'pink':   '#f15c80',
    }

    return trans_table[color]

def enhance_color(color, h=1, l=1, s=1):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))

    h, l, s = h * c[0], l * c[1], s * c[2]
    h, l, s = [max(min(x, 1), 0) for x in [h, l, s]]

    return colorsys.hls_to_rgb(h, l, s)
