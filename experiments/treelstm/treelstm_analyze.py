import numpy as np

from validate_config import validate
from common import invoke_main, write_status, write_json, render_exception
from analysis_util import trials_stat_summary, add_detailed_summary

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    frameworks = config['frameworks']
    methods = config['relay_methods']
    devices = config['devices']
    num_reps = config['n_inputs']
    datasets = config['datasets']

    listing_settings = {}

    base_fields = ['device']
    relay_fields = ['method']
    treelstm_fields = ['dataset', 'idx']

    if 'pt' in frameworks:
        listing_settings['Pytorch'] = ('pt', base_fields + treelstm_fields, {})

    if 'relay' in frameworks:
        fieldnames = base_fields + relay_fields + treelstm_fields
        if 'aot' in methods:
            listing_settings['Aot'] = ('relay', fieldnames, {'method': 'aot'})
        if 'intp' in methods:
            listing_settings['Intp'] = ('relay', fieldnames, {'method': 'intp'})


    # output averages on each network for each framework and each device
    ret = {}
    for dev in devices:
        ret[dev] = {}
        for listing, (framework, fieldnames, field_settings) in listing_settings.items():
            ret[dev][listing] = {}
            field_values = {
                'device': dev
            }
            for extra_field, value in field_settings.items():
                field_values[extra_field] = value

            # compute mean over datasets
            dataset_means = []
            for (dataset, _) in datasets:
                field_values['dataset'] = dataset
                summary, success, msg = trials_stat_summary(data_dir, framework, 'treelstm', num_reps,
                                                            fieldnames, field_values)
                if not success:
                    write_status(output_dir, False, msg)
                    return 1
                dataset_means.append(summary['mean'])
                add_detailed_summary(ret, summary, dev, listing, dataset)
            ret[dev][listing] = np.mean(dataset_means)

    write_json(output_dir, 'data.json', ret)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
