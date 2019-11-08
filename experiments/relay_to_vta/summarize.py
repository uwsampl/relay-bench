from decimal import Decimal

from validate_config import validate
from common import invoke_main, write_status, write_summary, sort_data

SIM_TARGETS = {'sim', 'tsim'}
PHYS_TARGETS = {'pynq'}
METADATA_KEYS = {'timestamp', 'tvm_hash',
                 'start_time', 'end_time', 'time_delta'}

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    all_data = sort_data(data_dir)
    most_recent = all_data[-1]
    most_recent = {k: v for (k, v) in most_recent.items() if k not in METADATA_KEYS}
    summary = ''

    for (model, targets) in most_recent.items():
        # simulated target summary
        sim_targets = {target: targets[target] for target in targets if target in SIM_TARGETS}
        for (target, devices) in sim_targets.items():
            for (device, stats) in devices.items():
                summary += '_Stats on ({}, {}, {}) & _\n'.format(model, target.upper(), device.upper())
                for (stat, val) in stats.items():
                    summary += '{}: {:.2E}\n'.format(stat, Decimal(val))
        # physical target summary
        phys_targets = {target: v for (target, v) in targets.items() if target in PHYS_TARGETS}
        for (target, devices) in phys_targets.items():
            for (device, mean_time) in devices.items():
                summary += 'Time on ({}, {}, {}): {:.2f}\n'.format(
                        model, target.upper(), device.upper(), mean_time)

    write_summary(output_dir, config['title'], summary)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
