from validate_config import validate
from exp_templates import (common_trial_params, common_early_exit,
                           run_template)

from language_data import N_LETTERS
from relay_rnn import samples, RNNCellOnly, RNNLoop


def rnn_setup(device, configuration, method, hidden_size, lang, letters):
    gpu = (device == 'gpu')
    cell_only = (configuration == 'cell')
    aot = (method == 'aot')

    net = RNNCellOnly if cell_only else RNNLoop
    init_net = net(aot, gpu, N_LETTERS, hidden_size, N_LETTERS)

    if cell_only:
        return [lambda: samples(init_net, lang, letters)]
    return [lambda: init_net.samples(lang, letters)]


def rnn_trial(thunk):
    return thunk()


def rnn_teardown(thunk):
    pass


if __name__ == '__main__':
    run_template(validate_config=validate,
                 check_early_exit=common_early_exit({'frameworks': 'relay'}),
                 gen_trial_params=common_trial_params(
                     'relay', 'char_rnn',
                     rnn_trial, rnn_setup, rnn_teardown,
                     ['device', 'configuration', 'method',
                      'hidden_size', 'language', 'input'],
                     ['devices', 'relay_configs', 'relay_methods',
                      'hidden_sizes', 'languages', 'inputs']))
