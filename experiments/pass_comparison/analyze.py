import itertools

from validate_config import validate
from common import invoke_main, write_status, write_json
from analysis_util import trials_stat_summary, add_detailed_summary

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    devices = config['devices']
    networks = config['networks']
    num_reps = config['n_inputs']
    batch_size = list(config['batch_sizes'])[0]
    passes = [';'.join([str(pass_spec[0]), '|'.join(pass_spec[1])])
              for pass_spec in config['passes']]

    # output averages on each network for each opt level,
    # pass specification, and device
    ret = {}
    for dev in devices:
        ret[dev] = {}
        for pass_str in passes:
            ret[dev][pass_str] = {}
            for network in networks:
                summary, success, msg = trials_stat_summary(data_dir, 'relay', 'pass_comparison', num_reps,
                                                            ['network', 'device', 'batch_size', 'pass'],
                                                            {'batch_size': batch_size,
                                                             'network': network,
                                                             'device': dev,
                                                             'pass': pass_str})
                if not success:
                    write_status(output_dir, False, msg)
                    return
                ret[dev][pass_str][network] = summary['mean']
                add_detailed_summary(ret, summary, dev, pass_str, network)

    write_json(output_dir, 'data.json', ret)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
