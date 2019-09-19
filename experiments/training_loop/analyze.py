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
    devices = config['devices']
    num_reps = config['n_inputs']
    num_classes = list(config['num_classes'])[0]
    batch_size = list(config['batch_sizes'])[0]
    epochs = config['epochs']

    listing_settings = {
        'Relay': 'relay',
        'Keras': 'keras'
    }

    fieldnames = ['device', 'batch_size', 'num_classes', 'epochs']

    # output averages on each network for each framework and each device
    ret = {}
    for dev in devices:
        ret[dev] = {}
        for listing, framework in listing_settings.items():
            ret[dev][listing] = {}
            for epoch_count in epochs:
                field_values = {
                    'device': dev,
                    'batch_size': batch_size,
                    'num_classes': num_classes,
                    'epochs': epoch_count
                }

                summary, success, msg = trials_stat_summary(data_dir, framework, 'training_loop', num_reps,
                                                            fieldnames, field_values)
                if not success:
                    write_status(output_dir, False, msg)
                    return 1
                ret[dev][listing][epoch_count] = summary['mean']
                add_detailed_summary(ret, summary, dev, listing, epoch_count)

    write_json(output_dir, 'data.json', ret)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
