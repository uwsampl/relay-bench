"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config
from config_util import check_config, bool_cond, non_negative_cond

VALID_PASSES = {
    '',
    'FoldScaleAxis',
    'BackwardFoldScaleAxis',
    'ForwardFoldScaleAxis',
    'FuseOps',
    'FoldConstant',
    'CombineParallelConv2D',
    'AlterOpLayout',
    'EliminateCommonSubexpr',
    'CanonicalizeCast',
    'CanonicalizeOps'
}

def valid_pass_spec_cond():
    def check_pass_specification(passes):
        if not isinstance(passes, tuple):
            return False
        if len(passes) != 2:
            return False
        opt_level = passes[0]
        pass_list = passes[1]

        if not isinstance(opt_level, int) or (opt_level < 0 or opt_level > 3):
            return False

        for pass_name in pass_list:
            if not isinstance(pass_name, str):
                return False
            if pass_name not in VALID_PASSES:
                return False
        return True

    return (check_pass_specification, 'is a valid pair of (opt_level, list of Relay passes)')

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
            opt_level = 0
            pass_list = ()
            if isinstance(pass_entry, list):
                if len(pass_entry) > 0:
                    opt_level = pass_entry[0]
                if len(pass_entry) > 1:
                    list_entry = pass_entry[1]
                    if isinstance(list_entry, str) and list_entry != '':
                        pass_list = (list_entry,)
                    if isinstance(list_entry, list):
                        pass_list = tuple(list_entry)
            new_entry = (opt_level, pass_list)
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
            'passes': {(3, pass_name) for pass_name in VALID_PASSES},
            'set_seed': False,
            'seed': 0
        },
        {
            'devices': {'cpu', 'gpu'},
            'networks': {'resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16'}
        },
        {
            'passes': valid_pass_spec_cond(),
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'batch_sizes': non_negative_cond(),
            'set_seed': bool_cond(),
            'seed': non_negative_cond()
        }
    )
