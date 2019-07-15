"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config
import string

from config_util import check_config, non_negative_cond, string_cond

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
        },
        {
            'devices': {'cpu'},
            'frameworks': {'pt', 'relay'},
            'relay_methods': {'intp', 'aot'},
            'relay_configs': {'loop', 'cell'},
            'languages': {'Arabic', 'Chinese', 'Czech', 'Dutch',
                          'English', 'French', 'German', 'Greek',
                          'Irish', 'Italian', 'Japanese', 'Korean',
                          'Polish', 'Portuguese', 'Russian', 'Scottish',
                          'Spanish', 'Vietnamese'}
        },
        {
            'hidden_sizes': non_negative_cond(),
            'inputs': string_cond(),
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond()
        }
    )
