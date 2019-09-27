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
            'dry_run': 3,
            'devices': {'gpu'},
            'frameworks': {'pt'},
            'datasets': {'mnist'},
            'models': {'mlp'},
            'batch_size': 32,
            'epochs': 5,
            'reps': 3,
            'set_seed': False,
            'seed': 0
        },
        {
            'devices': {'gpu'},
            'frameworks': {'pt'},
            'models': {'mlp'}
        },
        {
            'batch_size': non_negative_cond(),
            'epochs': non_negative_cond(),
            'dry_run': non_negative_cond(),
            'reps': non_negative_cond(),
            'set_seed': bool_cond(),
            'seed': non_negative_cond()
        }
    )

    return ret, msg
