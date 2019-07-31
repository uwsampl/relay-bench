"""
Common Python utilities for interacting with the dashboard infra.
"""
import datetime
import json
import logging
import os
import sys

def check_file_exists(dirname, filename):
    dirname = os.path.expanduser(dirname)
    full_name = os.path.join(dirname, filename)
    return os.path.isfile(full_name)


def idemp_mkdir(dirname):
    '''Creates a directory in an idempotent fashion.'''
    dirname = os.path.expanduser(dirname)
    os.makedirs(dirname, exist_ok=True)


def prepare_out_file(dirname, filename):
    dirname = os.path.expanduser(dirname)
    full_name = os.path.join(dirname, filename)
    if not check_file_exists(dirname, filename):
        os.makedirs(os.path.dirname(full_name), exist_ok=True)
    return full_name


def read_json(dirname, filename):
    dirname = os.path.expanduser(dirname)
    with open(os.path.join(dirname, filename)) as json_file:
        data = json.load(json_file)
        return data


def write_json(dirname, filename, obj):
    filename = prepare_out_file(dirname, filename)
    with open(filename, 'w') as outfile:
        json.dump(obj, outfile)


def read_config(dirname):
    return read_json(dirname, 'config.json')


def write_status(output_dir, success, message):
    write_json(output_dir, 'status.json', {
        'success': success,
        'message': message
    })


def write_summary(output_dir, title, value):
    write_json(output_dir, 'summary.json', {
        'title': title,
        'value': value
    })


def parse_timestamp(data):
    return datetime.datetime.strptime(data['timestamp'], '%m-%d-%Y-%H%M')


def sort_data(data_dir):
    '''Sorts all data files in the given directory by timestamp.'''
    data_dir = os.path.expanduser(data_dir)
    all_data = []
    for _, _, files in os.walk(data_dir):
        for name in files:
            data = read_json(data_dir, name)
            all_data.append(data)
    return sorted(all_data, key=parse_timestamp)


def gather_stats(sorted_data, fields):
    '''
    Expects input in the form of a list of data objects with timestamp
    fields (like those returned by sort_data).
    For each entry, this looks up entry[field[0]][field[1]]...
    for all entries that have all the fields, skipping those that
    don't. Returns a pair (list of entry values,
    list of corresponding entry timestamps)
    '''
    stats = []
    times = []
    for entry in sorted_data:
        stat = entry
        not_present = False
        for field in fields:
            if field not in stat:
                not_present = True
                break
            stat = stat[field]
        if not_present:
            continue
        times.append(parse_timestamp(entry))
        stats.append(stat)
    return (stats, times)


def traverse_fields(entry, omit_timestamp=True):
    '''
    Returns a list of sets of nested fields (one set per level of nesting)
    of a JSON data entry produced by a benchmark analysis script.
    Ignores the dashboard-appended 'timestamp' field at the top level by default.
    '''
    level_fields = {field for field in entry.keys()
                    if not (omit_timestamp and field == 'timestamp')}
    values_to_check = [value for value in entry.values()
                       if isinstance(value, dict)]

    tail = []
    max_len = 0
    for value in values_to_check:
        next_fields = traverse_fields(value)
        tail.append(next_fields)
        if len(next_fields) > max_len:
            max_len = len(next_fields)

    # combine all the field lists (union of each level's sets)
    final_tail = []
    for i in range(max_len):
        u = set({})
        final_tail.append(u.union(*[fields_list[i]
                                    for fields_list in tail
                                    if len(fields_list) > i]))

    return [level_fields] + final_tail



def render_exception(e):
    return logging.Formatter.formatException(e, sys.exc_info())
