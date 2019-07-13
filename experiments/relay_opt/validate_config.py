"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config

def non_negative(value):
    return isinstance(value, int) and value >= 0

def validate(config_dir):
    """
    Reads config.json in the config_dir and prepopulates with default values.
    Ensures that all configured values are of the appropriate types.

    Returns (config, message to report if error). Returns None if something
    is wrong with the config it read.
    """
    config = read_config(config_dir)
    ret = {
        'devices': {'cpu', 'gpu'},
        'dry_run': 8,
        'n_inputs': 3,
        'n_times_per_input': 10,
        'batch_sizes': [1],
        'opt_levels': [0,1,2,3],
        'networks': ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    }

    if 'devices' in config:
        devs = config['devices']
        if not devs:
            return None, 'No devs specified'
        acceptable_devs = {'cpu', 'gpu'}
        for dev in devs:
            if dev not in acceptable_devs:
                return None, 'Invalid dev specified: {}'.format(dev)
        ret['devices'] = set(devs)

    if 'dry_run' in config:
        if not non_negative(config['dry_run']):
            return None, 'Dry run must be non-negative'
        ret['dry_run'] = config['dry_run']

    if 'n_inputs' in config:
        if not non_negative(config['n_inputs']):
            return None, 'n_inputs must be non-negative'
        ret['n_inputs'] = config['n_inputs']

    if 'n_times_per_input' in config:
        value = config['n_times_per_input']
        if not non_negative(value):
            return None, 'Times per input must be non-negative'
        ret['n_times_per_input'] = value

    if 'batch_sizes' in config:
        if not config['batch_sizes']:
            return None, 'No batch sizes specified'
        for size in config['batch_sizes']:
            if not non_negative(size):
                return None, 'Batch sizes must all be non-negative'
        ret['batch_sizes'] = config['batch_sizes']

    if 'opt_levels' in config:
        if not config['opt_levels']:
            return None, 'No opt levels specified'
        for level in config['opt_levels']:
            if not non_negative(level):
                return None, 'All opt levels must be non-negative'
        ret['opt_levels'] = config['opt_levels']

    if 'networks' in config:
        if not config['networks']:
            return None, 'No networks specified'
        ret['networks'] = config['networks']

    return ret, ''
