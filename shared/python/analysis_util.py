"""
Utility functions for parsing data files. Especially designed for the CSVs produced
by trial_util.py
"""
import csv
import datetime
import os

import numpy as np

from common import render_exception

def lookup_data_file(data_prefix, filename):
    full_name = os.path.join(data_prefix, filename)
    if not os.path.exists(full_name):
        raise Exception('Could not find "{}"'.format(filename))
    return full_name


def mean_of_means(data, trait_name, trait_values, average_key='time', is_numeric=False):
    """
    Expects the data to be a list of dicts.
    For each entry r in the data and each value v in trait_values,
        takes the mean of all r[average_key] where r[trait_name] == v.
    Returns the mean of all the means previously computed.

    is_numeric: Whether the trait is numeric or not (expects string by default)
    """
    means = []
    for value in trait_values:
        def filter_func(r):
            if is_numeric:
                return int(r[trait_name]) == value
            return r[trait_name] == value
        mean = np.mean(list(map(lambda r: float(r[average_key]),
                                filter(filter_func, data))))
        means.append(mean)
    return np.mean(means)


def average_across_reps(data, num_reps):
    return mean_of_means(data, 'rep', range(num_reps), is_numeric=True)


def trials_average_time(data_dir, framework, task_name, num_reps, parameter_names, params_to_match):
    """
    Computes average for all trials on the specified framework and dask (looking for the CSV
    file in the given directory) across reps where all the specified parameters match.

    Returns (computed mean, success, message if failure)
    """
    filename = lookup_data_file(data_dir, '{}-{}.csv'.format(framework, task_name))
    with open(filename, newline='') as csvfile:
        # even though it seems redundant, parameter names does
        # need to be a separate arg because *order matters*
        # whereas it doesn't in a dict
        fieldnames = parameter_names + ['rep', 'run', 'time']
        reader = csv.DictReader(csvfile, fieldnames)

        def filter_func(row):
            for (name, value) in params_to_match.items():
                comp = value
                if not isinstance(value, str):
                    comp = str(value)
                if row[name] != comp:
                    return False
            return True

        try:
            return (average_across_reps(list(filter(filter_func, reader)), num_reps),
                    True, 'success')
        except Exception as e:
            return (-1,
                    False,
                    'Encountered exception on {}, {} using params {}:\n{}'.format(
                        framework, task_name, params_to_match,
                        render_exception(e)))
