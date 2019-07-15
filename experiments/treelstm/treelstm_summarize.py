import argparse

from validate_config import validate
from common import (write_status, write_summary,
                    parse_timestamp, sort_data, render_exception)

def treelstm_text_summary(data, devs):
    if not devs:
        return ''
    data_by_dev = {dev: data['treelstm-{}'.format(dev)] for dev in devs}
    for dev in devs:
        ret = '_Times on {}_\n'.format(dev.upper())
        for (setting, time) in data_by_dev[dev].items():
            ret += '{}: {:.3f}\n'.format(setting, time*1e3)
    return ret


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    devs = config['devices']

    # TODO: include some kind of comparison
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    try:
        summary = treelstm_text_summary(most_recent, devs)
        write_summary(output_dir, 'Treelstm Comparisons', summary)
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
