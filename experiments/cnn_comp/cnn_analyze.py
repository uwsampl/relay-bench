import argparse

from validate_config import validate
from common import write_status, write_json
from analysis_util import trials_average_time

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    frameworks = config['frameworks']
    devices = config['devices']
    networks = config['networks']
    num_reps = config['n_inputs']
    batch_size = list(config['batch_sizes'])[0]

    nice_name = {
        'relay': 'Relay',
        'tf': 'TensorFlow',
        'pt': 'Pytorch',
        'mxnet': 'MxNet',
        'nnvm': 'NNVM'
    }

    default_fields = ['network', 'device', 'batch_size']

    extra_fields = {
        'relay': {'opt_level': config['relay_opt']},
        'nnvm': {'opt_level': config['nnvm_opt']},
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

    # output averages on each network for each framework and each device
    ret = {}
    for dev in devices:
        dev_field = 'cnn-{}'.format(dev)
        ret[dev_field] = {}
        for listing, (framework, field_settings) in listing_settings.items():
            ret[dev_field][listing] = {}
            for network in networks:
                fields = default_fields + [key for key in field_settings.keys()]
                field_values = {
                    'network': network,
                    'device': dev,
                    'batch_size': batch_size
                }
                for extra_field, value in field_settings.items():
                    field_values[extra_field] = value

                mean, success, msg = trials_average_time(data_dir, framework, 'cnn_comp', num_reps,
                                                         fields, field_values)
                if not success:
                    write_status(output_dir, False, msg)
                    return
                ret[dev_field][listing][network] = mean

    write_json(output_dir, 'data.json', ret)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.config_dir, args.output_dir)
