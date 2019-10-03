from validate_config import validate
from exp_templates import analysis_template


def generate_listing_settings(config):
    listing_settings = {}

    frameworks = config['frameworks']
    methods = config['relay_methods']

    if 'mxnet' in frameworks:
        listing_settings['MxNet'] = ('mxnet', {})

    if 'relay' in frameworks:
        if 'aot' in methods:
            listing_settings['Aot'] = ('relay', {'method': 'aot'})
        if 'intp' in methods:
            listing_settings['Intp'] = ('relay', {'method': 'intp'})

    return listing_settings


def generate_data_query(config, dev, network, settings):
    fw = settings[0]
    special_fields = settings[1]

    num_reps = config['n_inputs']

    fields = ['device', 'network']
    if fw == 'relay':
        fields = ['device', 'method', 'network']

    field_values = {'device': dev, 'network': network, **special_fields}

    return [fw, 'gluon_rnns', num_reps, fields, field_values]


if __name__ == '__main__':
    analysis_template(validate, generate_listing_settings,
                      generate_data_query, use_networks=True)
