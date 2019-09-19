from validate_config import validate
from common import invoke_main, write_status, write_json
from analysis_util import trials_stat_summary, add_detailed_summary

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    opt_levels = sorted(list(config['opt_levels']))
    devices = config['devices']
    networks = config['networks']
    num_reps = config['n_inputs']
    batch_size = list(config['batch_sizes'])[0]

    # output averages on each network for each opt level and each device
    ret = {}
    for dev in devices:
        ret[dev] = {}
        for opt_level in opt_levels:
            level_field = 'O{}'.format(opt_level)
            ret[dev][level_field] = {}
            for network in networks:
                summary, success, msg = trials_stat_summary(data_dir, 'relay', 'opt_comparison', num_reps,
                                                            ['network', 'device', 'batch_size', 'opt_level'],
                                                            {'batch_size': batch_size,
                                                             'network': network,
                                                             'device': dev,
                                                             'opt_level': opt_level})
                if not success:
                    write_status(output_dir, False, msg)
                    return
                ret[dev][level_field][network] = summary['mean']
                add_detailed_summary(ret, summary, dev, level_field, network)

    write_json(output_dir, 'data.json', ret)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
