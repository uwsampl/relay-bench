import argparse

from validate_config import validate
from common import (write_status, write_summary,
                    parse_timestamp, sort_data, render_exception)

def cnn_comp_text_summary(data, devs, networks):
    if not devs:
        return ''
    data_by_dev = {dev: data['cnn-{}'.format(dev)] for dev in devs}
    ret = 'Format: ({})\n'.format(', '.join(networks))
    for dev in devs:
        ret += '_Times on {}_\n'.format(dev.upper())
        for (framework, times) in data_by_dev[dev].items():
            ret += '{}: '.format(framework)
            ret += ', '.join(['{:.3f}'.format(time*1e3)
                              for (_, time) in times.items()])
            ret += '\n'
    return ret


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    devs = config['devices']
    networks = config['networks']

    # TODO: include some kind of comparison
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    try:
        summary = cnn_comp_text_summary(most_recent, devs, networks)
        write_summary(output_dir, 'CNN Framework Comparisons', summary)
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        return
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.config_dir, args.output_dir)
