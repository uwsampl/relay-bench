"""
Common Python utilities for interacting with the dashboard infra.
"""
import os

def check_file_exists(dirname, filename):
    full_name = os.path.join(data_prefix, filename)
    return os.isfile(full_name)


def prepare_out_file(dirname, filename):
    full_name = os.path.join(dirname, filename)
    if not check_file_exists(dirname, filename):
        os.makedirs(os.path.dirname(full_name))
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
    return write_json(output_dir, 'status.json', {
        'success': success,
        'message': message
    })
