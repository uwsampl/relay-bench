import enum
import sys

# TODO: Make config mutator?

def preprocess_frameworks(config):
    requested_frameworks = set(config['frameworks'])
    available_frameworks = set(get_frameworks())
    unavailable_frameworks = requested_frameworks - available_frameworks
    if len(unavailable_frameworks) != 0:
        raise Exception(f'frameworks {unavailable_frameworks} are unavailable for experiment {get_name()}')

def preprocess_config(config):
    get_frameworks_to_run(config)

def run(config):
    preprocess_config(config)
    num_layers = config['num_layers']
    num_trials = config['num_trials']
    return {
        'perf': [(420.0, 'seconds') for _ in range(num_trials)],
        'avg_perf': [(420.0, 'seconds') for _ in range(num_trials)],
        'mem_usage': (1.3, 'gb'),
        'num_layers': num_layers,
        'num_trials': num_trials,
    }

def get_frameworks():
    return ['relay', 'mxnet']

def get_config_params():
    return ['num_layers', 'num_trials']

def get_name():
    return 'DankNet'
