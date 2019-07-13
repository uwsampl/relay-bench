"""
Common Python utilities for interacting with the dashboard infra.
"""
import datetime
import json
import logging
import os
import sys

def check_file_exists(dirname, filename):
    full_name = os.path.join(dirname, filename)
    return os.path.isfile(full_name)


def prepare_out_file(dirname, filename):
    full_name = os.path.join(dirname, filename)
    if not check_file_exists(dirname, filename):
        os.makedirs(os.path.dirname(full_name), exist_ok=True)
    return full_name


def read_json(dirname, filename):
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
    all_data = []
    for _, _, files in os.walk(data_dir):
        for name in files:
            data = read_json(data_dir, name)
            all_data.append(data)
    return sorted(all_data, key=parse_timestamp)


def render_exception(e):
    return logging.Formatter.formatException(e, sys.exc_info())
