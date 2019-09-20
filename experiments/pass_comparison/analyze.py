from validate_config import validate
from exp_templates import analysis_template

def generate_listing_settings(config):
    passes = [';'.join([str(pass_spec[0]), '|'.join(pass_spec[1])])
              for pass_spec in config['passes']]
    return {pass_str: {'pass': pass_str} for pass_str in passes}


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

    return ['relay', 'pass_comparison', num_reps, fields, field_values]


if __name__ == '__main__':
    analysis_template(validate, generate_listing_settings,
                      generate_data_query, use_networks=True)
