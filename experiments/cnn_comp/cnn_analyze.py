from validate_config import validate
from exp_templates import analysis_template

def generate_listing_settings(config):
    frameworks = config['frameworks']

    nice_name = {
        'relay': 'Relay',
        'tf': 'TensorFlow',
        'pt': 'Pytorch',
        'mxnet': 'MxNet',
    }

    extra_fields = {
        'relay': {'opt_level': config['relay_opt']},
        'tf': {'enable_xla': False},
        'mxnet': {},
        'pt': {}
    }

    listing_settings = {
        nice_name[fw]: (fw, {field: value for field, value in extra_fields[fw].items()})
        for fw in frameworks
    }

    if 'tf' in frameworks and config['use_xla']:
        listing_settings['TF XLA'] = ('tf', {'enable_xla': True})

    return listing_settings


def generate_data_query(config, dev, network, settings):
    fw = settings[0]
    special_fields = settings[1]

    num_reps = config['n_inputs']
    batch_size = list(config['batch_sizes'])[0]

    fields = (['network', 'device', 'batch_size', *list(special_fields.keys())])
    field_values = {
        'network': network,
        'device': dev,
        'batch_size': batch_size,
        **special_fields
    }

    return [fw, 'cnn_comp', num_reps, fields, field_values]


if __name__ == '__main__':
    analysis_template(validate, generate_listing_settings,
                      generate_data_query, use_networks=True)
