"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config
from config_util import check_config, non_negative_cond

VALID_PASSES = {
    '',
    'FoldScaleAxis',
    'BackwardFoldScaleAxis',
    'ForwardFoldScaleAxis',
    'FuseOps',
    'FoldConstant',
    'CombineParallelConv2d',
    'AlterOpLayout',
    'EliminateCommonSubexpr',
    'CanonicalizeCast',
    'CanonicalizeOps'
}

def valid_pass_list_cond():
    def check_pass_list(passes):
        if not isinstance(passes, tuple):
            return False
        for pass_name in passes:
            if not isinstance(pass_name, str):
                return False
            if pass_name not in VALID_PASSES:
                return False
        return True

    return (check_pass_list, 'is a valid list of Relay passes')

def validate(config_dir):
    """
    Reads config.json in the config_dir and prepopulates with default values.
    Ensures that all configured values are of the appropriate types.

    Returns (config, message to report if error). Returns None if something
    is wrong with the config it read.
    """
    config = read_config(config_dir)

    # turn passes into tuples so they are hashable
    # turn strings into singletons for consistency
    if 'passes' in config:
        passes = []
        for pass_entry in config['passes']:
            new_entry = ()
            if isinstance(pass_entry, str) and pass_entry != '':
                new_entry = (pass_entry,)
            if isinstance(pass_entry, list):
                new_entry = tuple(pass_entry)
            passes.append(new_entry)
        config['passes'] = passes

    return check_config(
        config,
        {
            'devices': {'cpu', 'gpu'},
            'dry_run': 8,
            'n_inputs': 3,
            'n_times_per_input': 10,
            'batch_sizes': {1},
            'networks': {'resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16'},
            'passes': {(pass_name,) for pass_name in VALID_PASSES}
        },
        {
            'devices': {'cpu', 'gpu'},
            'networks': {'resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16'}
        },
        {
            'passes': valid_pass_list_cond(),
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'batch_sizes': non_negative_cond()
        }
    )
