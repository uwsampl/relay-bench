from validate_config import validate
from exp_templates import (common_trial_params, common_early_exit,
                           run_template)

from language_data import N_LETTERS
from pt_rnn import RNN, samples


def rnn_setup(device, hidden_size, lang, letters):
    rnn = RNN(N_LETTERS, hidden_size, N_LETTERS)
    return [lambda: samples(rnn, lang, letters)]


def rnn_trial(thunk):
    return thunk()


def rnn_teardown(thunk):
    pass


if __name__ == '__main__':
    run_template(validate_config=validate,
                 check_early_exit=common_early_exit({'frameworks': 'pt'}),
                 gen_trial_params=common_trial_params(
                     'pt', 'char_rnn',
                     rnn_trial, rnn_setup, rnn_teardown,
                     ['device', 'hidden_size', 'language', 'input'],
                     ['devices', 'hidden_sizes', 'languages', 'inputs']))
