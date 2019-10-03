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


def compute_summary_stats(data, trait_name, trait_values, average_key='time', is_numeric=False):
    """
    Expects the data to be a list of dicts.
    For each entry r in the data and each value v in trait_values,
    this function assembles the values of all fields
    r[trait_key] where r[trait_name] == v.
    Returns the mean, median, and std dev of all values in a
    dict with fields "mean", "median", and "std"

    is_numeric: Whether the trait is numeric or not (expects string by default)
    """
    vals = []
    for value in trait_values:
        def filter_func(r):
            if is_numeric:
                return int(r[trait_name]) == value
            return r[trait_name] == value
        vals += list(map(lambda r: float(r[average_key]),
                         filter(filter_func, data)))
    return {
        'mean': np.mean(vals),
        'median': np.median(vals),
        'std': np.std(vals)
    }


def summarize_over_reps(data, num_reps):
    return compute_summary_stats(data, 'rep', range(num_reps), is_numeric=True)


def obtain_data_rows(data_dir, framework, task_name, parameter_names, params_to_match):
    """
    Returns all data rows from the given framework from the
    given task where the specified parameters match.

    params_to_match as a dictionary {param names => value to match}
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
        return list(filter(filter_func, reader))


def trials_stat_summary(data_dir, framework, task_name,
                        num_reps, parameter_names, params_to_match):
    """
    Returns a full summary of statistics on the specified framework
    and task across all reps where the specified parameters match.

    Returns (summary, success, message)
    """
    try:
        data = obtain_data_rows(data_dir, framework, task_name, parameter_names, params_to_match)
        summary = summarize_over_reps(data, num_reps)
        return (summary, True, 'success')
    except Exception as e:
        return (-1, False,
                'Encountered exception on {}, {} using params {}:\n{}'.format(
                    framework, task_name, params_to_match,
                    render_exception(e)))


def add_detailed_summary(report, detailed_summary, *fields):
    """
    Nasty hack provided for including a more detailed statistical
    summary in analysis files. The main reason this is here is
    because old reports contained only the mean and the graphing,
    etc., made assumptions about the layout of records.
    Eventually the old records should be migrated.
    """
    current = report
    all_fields = ['detailed', *fields]
    for field in all_fields[:-1]:
        if field not in current:
            current[field] = {}
        current = current[field]
    current[all_fields[-1]] = detailed_summary
