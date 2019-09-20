from validate_config import validate
from exp_templates import analysis_template


def generate_listing_settings(config):
    listing_settings = {}

    frameworks = config['frameworks']
    methods = config['relay_methods']

    if 'pt' in frameworks:
        listing_settings['Pytorch'] = ('pt', {})

    if 'relay' in frameworks:
        if 'aot' in methods:
            listing_settings['Aot'] = ('relay', {'method': 'aot'})
        if 'intp' in methods:
            listing_settings['Intp'] = ('relay', {'method': 'intp'})

    return listing_settings


def generate_data_query(config, dev, settings):
    fw = settings[0]
    special_fields = settings[1]

    num_reps = config['n_inputs']

    base_fields = ['device']
    relay_fields = ['method']
    treelstm_fields = ['dataset', 'idx']

    fields = base_fields + treelstm_fields
    if fw == 'relay':
        fields = base_fields + relay_fields + treelstm_fields

    field_values = {'device': dev, **special_fields}

    return [fw, 'treelstm', num_reps, fields, field_values]


if __name__ == '__main__':
    analysis_template(validate, generate_listing_settings,
                      generate_data_query, use_networks=False)
