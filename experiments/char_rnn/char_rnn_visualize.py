import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from validate_config import validate
from common import (write_status, prepare_out_file, parse_timestamp,
                    sort_data, render_exception)
from plot_util import PlotType, PlotBuilder, generate_longitudinal_comparisons

def generate_char_rnn_comparison(title, filename, data, output_prefix=''):
    means = [measurement for (_, measurement) in data.items()]
    if not means:
        return

    comparison_dir = os.path.join(output_prefix, 'comparison')
    PlotBuilder().set_title(title) \
                 .set_x_label('Framework') \
                 .set_y_label('Time (ms)') \
                 .make(PlotType.BAR, data) \
                 .save(comparison_dir, filename)


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    devs = config['devices']

    # read in data, output graphs of most recent data, and output longitudinal graphs
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    try:
        generate_longitudinal_comparisons(all_data, output_dir)
        for dev in devs:
            generate_char_rnn_comparison('Char RNN Comparison on {}'.format(dev.upper()),
                                         'char_rnn-{}.png'.format(dev),
                                         most_recent[dev], output_dir)

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
