"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config
from config_util import check_config, non_negative_cond, string_cond

def validate(config_dir):
    """
    Reads config.json in the config_dir and prepopulates with default values.
    Ensures that all configured values are of the appropriate types.

    Returns (config, message to report if error). Returns None if something
    is wrong with the config it read.
    """
    config = read_config(config_dir)

    config, msg = check_config(
        config,
        {
            'models': ['resnet18_v1'],
            'devices': ['vta'],
            'targets': ['sim'],
            'n_times_per_input': 4,
            'num_reps': 3
        },
        {
            'models': {'resnet18_v1'},
            'devices': {'vta', 'arm_cpu'},
            'targets': {'pynq', 'sim', 'tsim'}
        },
        {
            'tracker_host': string_cond(),
            'tracker_port': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'num_reps': non_negative_cond()
        }
    )

    return config, msg
