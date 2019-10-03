"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config

from config_util import check_config, bool_cond, non_negative_cond

DATASET_MAX = {
    'dev': 500,
    'test': 4927,
    'train': 4500
}

def valid_dataset_cond():
    def check_dataset_pair(pair):
        dataset = pair[0]
        max_idx = pair[1]

        if not isinstance(dataset, str) or not isinstance(max_idx, int):
            return False

        if dataset not in {'dev', 'test', 'train'}:
            return False

        if max_idx < 0:
            return False

        if max_idx > DATASET_MAX[dataset]:
            return False

        return True

    return (check_dataset_pair, 'is a valid pair of dataset and max index')

def validate(config_dir):
    """
    Reads config.json in the config_dir and prepopulates with default values.
    Ensures that all configured values are of the appropriate types.

    Returns (config, message to report if error). Returns None if something
    is wrong with the config it read.
    """
    config = read_config(config_dir)
    # need to turn into tuples so they are hashable
    config['datasets'] = [tuple(pair) for pair in config['datasets']]
    ret, msg = check_config(
        config,
        {
            'dry_run': 8,
            'n_inputs': 1,
            'n_times_per_input': 100,
            'devices': {'cpu'},
            'frameworks': {'pt', 'relay'},
            'relay_methods': {'intp', 'aot'},
            'set_seed': False,
            'seed': 0
        },
        {
            'devices': {'cpu'},
            'frameworks': {'pt', 'relay'},
            'relay_methods': {'intp', 'aot'}
        },
        {
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'datasets': valid_dataset_cond(),
            'set_seed': bool_cond(),
            'seed': non_negative_cond()
        }
    )

    # also must ensure that if relay is enabled that a method is specified
    if ret is not None and 'relay' in ret['frameworks']:
        if not ret['relay_methods']:
            return None, 'Relay is enabled but no method is specified'

    return ret, msg
