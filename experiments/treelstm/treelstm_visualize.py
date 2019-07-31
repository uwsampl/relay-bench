import argparse
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import numpy as np

from validate_config import validate
from common import (write_status, prepare_out_file, parse_timestamp,
                    sort_data, render_exception)
from plot_util import format_ms, generate_longitudinal_comparisons


def generate_treelstm_comparison(title, filename, data, output_prefix=''):
    fig, ax = plt.subplots()
    format_ms(ax)

    comparison_dir = os.path.join(output_prefix, 'comparison')

    means = [measurement for (_, measurement) in data.items()]
    if not means:
        return

    settings = np.arange(len(data.items()))
    plt.bar(settings, means)
    plt.xticks(settings, [name for (name, _) in data.items()])
    plt.title(title)
    plt.xlabel('Framework')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    outfile = prepare_out_file(comparison_dir, filename)
    plt.savefig(outfile)
    plt.close()


def generate_longitudinal_comparisons(sorted_data, dev, output_prefix=''):
    if not sorted_data:
        return

    longitudinal_dir = os.path.join(output_prefix, 'longitudinal')

    times = [parse_timestamp(entry) for entry in sorted_data]
    most_recent = sorted_data[-1][dev]
    for (setting, time) in most_recent.items():
        stats = [entry[dev][setting] for entry in sorted_data]

        fig, ax = plt.subplots()
        format_ms(ax)
        plt.plot(times, stats)
        plt.title('{} on {} over Time'.format(setting, dev))
        filename = 'longitudinal-{}-{}.png'.format(setting, dev)
        plt.xlabel('Date of Run')
        plt.ylabel('Time (ms)')
        plt.yscale('log')
        plt.gcf().autofmt_xdate()
        outfile = prepare_out_file(longitudinal_dir, filename)
        plt.savefig(outfile)
        plt.close()


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
            generate_treelstm_comparison('TreeLSTM Comparison on {}'.format(dev.upper()),
                                         'treelstm-{}.png'.format(dev),
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
