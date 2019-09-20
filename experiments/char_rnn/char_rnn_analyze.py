from validate_config import validate
from exp_templates import analysis_template


def generate_listing_settings(config):
    listing_settings = {}

    frameworks = config['frameworks']
    methods = config['relay_methods']
    configurations = config['relay_configs']

    if 'pt' in frameworks:
        listing_settings['Pytorch'] = ('pt', {})

    if 'relay' in frameworks:
        if 'aot' in methods and 'loop' in configurations:
            listing_settings['Aot'] = (
                'relay', {'method': 'aot', 'configuration': 'loop'})
        if 'intp' in methods:
            for conf in configurations:
                name = 'Intp {}'.format(conf.capitalize())
                listing_settings[name] = (
                    'relay', {'method': 'intp', 'configuration': conf})

    return listing_settings


def generate_data_query(config, dev, settings):
    fw = settings[0]
    special_fields = settings[1]

    num_reps = config['n_inputs']
    inp = list(config['inputs'])[0]
    hidden_size = list(config['hidden_sizes'])[0]

    base_fields = ['device']
    relay_fields = ['configuration', 'method']
    char_rnn_fields = ['hidden_size', 'language', 'input']

    fields = base_fields + char_rnn_fields
    if fw == 'relay':
        fields = base_fields + relay_fields + char_rnn_fields

    field_values = {
        'device': dev,
        'hidden_size': hidden_size,
        'input': inp,
        **special_fields
    }

    return [fw, 'char_rnn', num_reps, fields, field_values]


if __name__ == '__main__':
    analysis_template(validate, generate_listing_settings,
                      generate_data_query, use_networks=False)
