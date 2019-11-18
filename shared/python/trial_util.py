import csv
from itertools import product
import os
import random
import time

import numpy as np
import mxnet as mx
import tensorflow as tf
import torch as pt

from common import render_exception


def set_seed(seed):
    # cover our bases: different frameworks and libraries
    # need to have their seeds set
    np.random.seed(seed)
    mx.random.seed(seed)
    tf.set_random_seed(seed)
    pt.manual_seed(seed)
    random.seed(seed)


def configure_seed(config):
    """
    Convenience for experiment scripts: Takes an experiment config
    and sets the seed if specified.

    Assumes that the config has a boolean field called 'set_seed'
    and an integer field called 'seed' for determining whether to
    set the seed and the value to use.
    """
    if config['set_seed']:
        set_seed(config['seed'])


def _write_row(writer, fieldnames, fields):
    record = {}
    for i in range(len(fieldnames)):
        record[fieldnames[i]] = fields[i]
    writer.writerow(record)


def _score_loop(num, trial, trial_args, setup_args, n_times, dry_run, writer, fieldnames):
    for i in range(dry_run + n_times):
        if i == dry_run:
            tic = time.time()
        start = time.time()
        out = trial(*trial_args)
        end = time.time()
        if i >= dry_run:
            _write_row(writer, fieldnames, setup_args + [num, i, end - start])

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
                narg = list(args)
                if narg and narg[0] == 'cpu':
                    pid = os.getpid()
                    print(f'Pinning PID: {pid}')
                    os.system(f'taskset -p 0x000000ff {pid}')
                costs = []
                for t in range(n_input):
                    score = 0.0
                    try:
                        trial_args = trial_setup(*args)
                        score = _score_loop(t, trial, trial_args, list(narg),
                                            times_per_input, dry_run,
                                            writer, fieldnames)
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

def _array2str_round(x, decimal=6):
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
