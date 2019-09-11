import numpy as np

from validate_config import validate
from common import write_status, write_json, render_exception, invoke_main
from analysis_util import trials_average_time, mean_of_means

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    frameworks = config['frameworks']
    methods = config['relay_methods']
    configurations = config['relay_configs']
    devices = config['devices']
    languages = config['languages']
    num_reps = config['n_inputs']
    inp = list(config['inputs'])[0]
    hidden_size = list(config['hidden_sizes'])[0]

    listing_settings = {}

    base_fields = ['device']
    relay_fields = ['configuration', 'method']
    char_rnn_fields = ['hidden_size', 'language', 'input']

    if 'pt' in frameworks:
        listing_settings['Pytorch'] = ('pt', base_fields + char_rnn_fields, {})

    if 'relay' in frameworks:
        fieldnames = base_fields + relay_fields + char_rnn_fields
        if 'aot' in methods and 'loop' in configurations:
            listing_settings['Aot'] = ('relay', fieldnames,
                                       {'method': 'aot', 'configuration': 'loop'})
        if 'intp' in methods:
            for conf in configurations:
                listing_settings['Intp {}'.format(conf.capitalize())] = ('relay',
                                                            fieldnames, {'method': 'intp', 'configuration': conf})


    # output averages on each network for each framework and each device
    ret = {}
    for dev in devices:
        ret[dev] = {}
        for listing, (framework, fieldnames, field_settings) in listing_settings.items():
            ret[dev][listing] = {}
            field_values = {
                'device': dev,
                'hidden_size': hidden_size,
                'input': inp
            }
            for extra_field, value in field_settings.items():
                field_values[extra_field] = value

            # compute mean over languages
            language_means = []
            for language in languages:
                field_values['language'] = language
                mean, success, msg = trials_average_time(data_dir, framework, 'char_rnn', num_reps,
                                                         fieldnames, field_values)
                if not success:
                    write_status(output_dir, False, msg)
                    return
                language_means.append(mean)
            ret[dev][listing] = np.mean(language_means)

    write_json(output_dir, 'data.json', ret)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
