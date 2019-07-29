import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from validate_config import validate
from common import (write_status, prepare_out_file, parse_timestamp,
                    sort_data, render_exception)
from plot_util import PlotType, make_plot

def generate_char_rnn_comparison(title, filename, data, output_prefix=''):
    means = [measurement for (_, measurement) in data.items()]
    if not means:
        return

    comparison_dir = os.path.join(output_prefix, 'comparison')
    x_label = 'Framework'
    y_label = 'Time (ms)'
    settings = np.arange(len(data.items()))
    x_tick_labels = [name for (name, _) in data.items()]
    make_plot(PlotType.BAR, title, x_label, y_label,
              settings, means,
              comparison_dir, filename,
              x_tick_labels=x_tick_labels)


def generate_longitudinal_comparisons(sorted_data, dev, output_prefix=''):
    if not sorted_data:
        return

    longitudinal_dir = os.path.join(output_prefix, 'longitudinal')

    times = [parse_timestamp(entry) for entry in sorted_data]
    most_recent = sorted_data[-1][dev]
    for (setting, time) in most_recent.items():
        stats = [entry[dev][setting] for entry in sorted_data]

        title = '{} on {} over Time'.format(setting, dev)
        filename = 'longitudinal-{}-{}.png'.format(setting, dev)
        x_label = 'Date of Run'
        y_label = 'Time (ms)'

        make_plot(PlotType.LINE, title, x_label, y_label,
                  times, stats,
                  longitudinal_dir, filename)


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    devs = config['devices']

    # read in data, output graphs of most recent data, and output longitudinal graphs
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    for dev in devs:
        try:
            generate_char_rnn_comparison('Char RNN Comparison on {}'.format(dev.upper()),
                                         'char_rnn-{}.png'.format(dev),
                                         most_recent[dev], output_dir)
            # TODO: do a better job with longitudinal comparisons
            generate_longitudinal_comparisons(all_data, dev, output_dir)
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
