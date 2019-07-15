"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config
from config_util import check_config, non_negative_cond

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
            'opt_levels': {0,1,2,3},
            'networks': {'resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16'}
        },
        {
            'devices': {'cpu', 'gpu'},
            'opt_levels': {0,1,2,3},
            'networks': {'resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16'}
        },
        {
            'opt_levels': non_negative_cond(),
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'batch_sizes': non_negative_cond()
        }
    )
