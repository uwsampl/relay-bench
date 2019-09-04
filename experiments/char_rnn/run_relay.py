import argparse
import sys

from validate_config import validate
from common import write_status
from trial_util import run_trials, configure_seed

from language_data import N_LETTERS
from relay_rnn import samples, RNNCellOnly, RNNLoop

def rnn_setup(device, configuration, method, hidden_size, lang, letters):
    gpu = (device == 'gpu')
    cell_only = (configuration == 'cell')
    aot = (method == 'aot')

    net = RNNCellOnly if cell_only else RNNLoop
    init_net = net(aot, gpu, N_LETTERS, hidden_size, N_LETTERS)
    if cell_only:
        thunk = lambda: samples(init_net, lang, letters)
    else:
        thunk = lambda: init_net.samples(lang, letters)
    return [thunk]


def rnn_trial(thunk):
    return thunk()


def rnn_teardown(thunk):
    pass


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'relay' not in config['frameworks']:
        write_status(output_dir, True, 'Relay not run')
        sys.exit(0)

    configure_seed(config)

    success, msg = run_trials(
        'relay', 'char_rnn',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        rnn_trial, rnn_setup, rnn_teardown,
        ['device', 'configuration', 'method',
         'hidden_size', 'language', 'input'],
        [config['devices'], config['relay_configs'], config['relay_methods'],
         config['hidden_sizes'], config['languages'], config['inputs']],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir)
