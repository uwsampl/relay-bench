"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config

from config_util import check_config, non_negative_cond, bool_cond

def validate(config_dir):
    """
    Reads config.json in the config_dir and prepopulates with default values.
    Ensures that all configured values are of the appropriate types.

    Returns (config, message to report if error). Returns None if something
    is wrong with the config it read.
    """
    config = read_config(config_dir)
    return check_config(
        config,
        {
            'devices': {'cpu', 'gpu'},
            'dry_run': 8,
            'n_inputs': 3,
            'n_times_per_input': 10,
            'batch_sizes': {1},
            'relay_opt': 3,
            'use_xla': True,
            'frameworks': {'tf', 'pt', 'relay', 'mxnet'},
            'networks': {'resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16'},
            'set_seed': False,
            'seed': 0
        },
        {
            'devices':  {'cpu', 'gpu'},
            'frameworks': {'tf', 'pt', 'relay', 'mxnet'},
            'networks': {'resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16'}
        },
        {
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'batch_sizes': non_negative_cond(),
            'relay_opt': non_negative_cond(),
            'use_xla': bool_cond(),
            'set_seed': bool_cond(),
            'seed': non_negative_cond()
        }
    )
