"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config
import string

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
        'active': True,
        'dry_run': 8,
        'n_inputs': 1,
        'n_times_per_input': 100,
        'inputs': [string.ascii_uppercase],
        'devices': ['cpu'],
        'hidden_sizes': [16],
        'frameworks': {'pt', 'relay'},
        'relay_methods': {'intp', 'aot'},
        'relay_configs': {'loop', 'cell'},
        'languages': {'Arabic', 'Chinese', 'Czech', 'Dutch',
                      'English', 'French', 'German', 'Greek',
                      'Irish', 'Italian', 'Japanese', 'Korean',
                      'Polish', 'Portuguese', 'Russian', 'Scottish',
                      'Spanish', 'Vietnamese'}
    }

    if 'devices' in config:
        devs = config['devices']
        if not devs:
            return None, 'No devs specified'
        acceptable_devs = {'cpu'}
        for dev in devs:
            if dev not in acceptable_devs:
                return None, 'Invalid dev specified: {}'.format(dev)
        ret['devices'] = set(devs)

    if 'frameworks' in config:
        frameworks = config['frameworks']
        if not frameworks:
            return None, 'No frameworks specified'
        acceptable_fws = {'pt', 'relay'}
        for fw in frameworks:
            if fw not in acceptable_fws:
                return None, 'Invalid framework specified: {}'.format(fw)
        ret['frameworks'] = set(frameworks)

    if 'relay_methods' in config:
        methods = config['relay_methods']
        if not methods:
            return None, 'No Relay methods specified'
        acceptable_methods = {'aot', 'intp'}
        for method in methods:
            if method not in acceptable_methods:
                return None, 'Invalid Relay method specified: {}'.format(method)
        ret['relay_methods'] = set(methods)

    if 'relay_configs' in config:
        relay_configs = config['relay_configs']
        if not relay_configs:
            return None, 'No Relay configurations specified'
        acceptable_configs = {'cell', 'loop'}
        for relay_config in relay_configs:
            if relay_config not in acceptable_configs:
                return None, 'Invalid Relay configuration specified: {}'.format(relay_config)
        ret['relay_configs'] = set(relay_configs)

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

    if 'hidden_sizes' in config:
        if not config['hidden_sizes']:
            return None, 'No hidden sizes specified'
        for size in config['hidden_sizes']:
            if not non_negative(size):
                return None, 'Hidden sizes must all be non-negative'
        ret['hidden_sizes'] = config['hidden_sizes']

    if 'inputs' in config:
        inputs = config['inputs']
        if not inputs:
            return None, 'No network inputs specified'
        for input in inputs:
            if not isinstance(input, str):
                return None, 'Non-string input: {}'.format(input)
        ret['inputs'] = inputs

    if 'languages' in config:
        languages = config['languages']
        if not languages:
            return None, 'No languages specified'
        acceptable_languages = {'Arabic', 'Chinese', 'Czech', 'Dutch',
                                'English', 'French', 'German', 'Greek',
                                'Irish', 'Italian', 'Japanese', 'Korean',
                                'Polish', 'Portuguese', 'Russian', 'Scottish',
                                'Spanish', 'Vietnamese'}
        for language in languages:
            if language not in acceptable_languages:
                return None, 'Invalid language {} specified'.format(language)
        ret['languages'] = set(languages)

    return ret, ''
