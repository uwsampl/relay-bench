from validate_config import validate
from exp_templates import analysis_template

def generate_listing_settings(config):
    listing_settings = {
        'O{}'.format(i): ({'opt_level': i})
        for i in sorted(list(config['opt_levels']))
    }

    return listing_settings


def generate_data_query(config, dev, network, settings):
    num_reps = config['n_inputs']
    batch_size = list(config['batch_sizes'])[0]

    fields = (['network', 'device', 'batch_size', *list(settings.keys())])
    field_values = {
        'network': network,
        'device': dev,
        'batch_size': batch_size,
        **settings
    }

    return ['relay', 'opt_comparison', num_reps, fields, field_values]


if __name__ == '__main__':
    analysis_template(validate, generate_listing_settings,
                      generate_data_query, use_networks=True)
