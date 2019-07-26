import argparse
from decimal import Decimal

from validate_config import validate
from common import write_status, write_summary, sort_data

SIM_TARGETS = {'sim', 'tsim'}
PHYS_TARGETS = {'pynq'}

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    all_data = sort_data(data_dir)
    most_recent = all_data[-1]
    summary = ''

    for target in most_recent.keys() & SIM_TARGETS:
        summary = '_Stats on {}_\n'.format(target.upper())
        for (stat, val) in most_recent[target].items():
            summary += '{}: {:.2E}\n'.format(stat, Decimal(val))

    for target in most_recent.keys() & PHYS_TARGETS:
        data = most_recent[target]
        summary += '_Time on {}_: {} ± {}\n'.format(
            target.upper(), data['mean'], data['std_dev'])

    write_summary(output_dir, config['title'], summary)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.config_dir, args.output_dir)
