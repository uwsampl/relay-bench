import os
import time
import numpy as np
from itertools import product
import csv

from common import render_exception

def write_row(writer, fieldnames, fields):
    record = {}
    for i in range(len(fieldnames)):
        record[fieldnames[i]] = fields[i]
    writer.writerow(record)


def score_loop(num, trial, trial_args, setup_args, n_times, dry_run, writer, fieldnames):
    num_batches = 10
    batch_size = 1
    if n_times % num_batches == 0 and n_times > num_batches:
        batch_size = n_times / num_batches
    else:
        num_batches = n_times

    for i in range(dry_run):
        out = trial(*trial_args)
    tic = time.time()

    for i in range(num_batches):
        start = time.time()
        for j in range(int(batch_size)):
            out = trial(*trial_args)
        end = time.time()
        if i >= dry_run:
            write_row(writer, fieldnames, setup_args + [num, i, (end - start) / batch_size])

    final = time.time()
    return (final - tic) / n_times


def run_trials(method, task_name,
               dry_run, times_per_input, n_input,
               trial, trial_setup, trial_teardown,
               parameter_names, parameter_ranges,
               path_prefix = '',
               append_to_csv = False):
    try:
        filename = os.path.join(path_prefix, '{}-{}.csv'.format(method, task_name))
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        mode = 'a' if append_to_csv else 'w'
        with open(filename, mode, newline='') as csvfile:
            fieldnames = parameter_names + ['rep', 'run', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not append_to_csv:
                writer.writeheader()

            for args in product(*parameter_ranges):
                costs = []
                for t in range(n_input):
                    score = 0.0
                    try:
                        trial_args = trial_setup(*args)
                        score = score_loop(t, trial, trial_args, list(args), times_per_input, dry_run, writer, fieldnames)
                        trial_teardown(*trial_args)
                    except Exception as e:
                        # can provide more detailed summary if
                        # it happened inside a trial
                        return (False,
                                'Encountered exception in trial on inputs {}:\n'.format(args)
                                + render_exception(e))

                    if t != n_input - 1:
                        time.sleep(4)
                    costs.append(score)

                print(method, task_name, args, ["%.6f" % x for x in costs])
        return (True, 'success')
    except Exception as e:
        return (False, 'Encountered exception:\n' + render_exception(e))

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
