"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config
import string

from config_util import check_config, bool_cond, non_negative_cond, string_cond

def validate(config_dir):
    """
    Reads config.json in the config_dir and prepopulates with default values.
    Ensures that all configured values are of the appropriate types.

    Returns (config, message to report if error). Returns None if something
    is wrong with the config it read.
    """
    config = read_config(config_dir)
    ret, msg = check_config(
        config,
        {
            'dry_run': 2,
            'n_inputs': 1,
            'n_times_per_input': 3,
            'devices': ['cpu'],
            'frameworks': {'keras', 'relay'},
            'batch_sizes': [1],
            'num_classes': [10],
            'epochs': [20],
            'set_seed': False,
            'seed': 0
        },
        {
            'devices': {'cpu'},
            'frameworks': {'keras', 'relay'}
        },
        {
            'batch_sizes': non_negative_cond(),
            'num_classes': non_negative_cond(),
            'epochs': non_negative_cond(),
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'set_seed': bool_cond(),
            'seed': non_negative_cond()
        }
    )

    return ret, msg
