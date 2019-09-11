import sys

from validate_config import validate
from common import invoke_main, write_status
from trial_util import run_trials, configure_seed

from language_data import N_LETTERS
from pt_rnn import RNN, samples

def rnn_setup(device, hidden_size, lang, letters):
    rnn = RNN(N_LETTERS, hidden_size, N_LETTERS)
    return [lambda: samples(rnn, lang, letters)]


def rnn_trial(thunk):
    return thunk()


def rnn_teardown(thunk):
    pass


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'pt' not in config['frameworks']:
        write_status(output_dir, True, 'PT not run')
        sys.exit(0)

    configure_seed(config)

    success, msg = run_trials(
        'pt', 'char_rnn',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        rnn_trial, rnn_setup, rnn_teardown,
        ['device', 'hidden_size', 'language', 'input'],
        [config['devices'], config['hidden_sizes'],
         config['languages'], config['inputs']],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'output_dir')
