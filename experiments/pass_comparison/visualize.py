import argparse
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import numpy as np

from validate_config import validate
from common import (write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import format_ms, generate_longitudinal_comparisons


def generate_relay_opt_comparisons(title, filename, data, networks, output_prefix=''):
    fig, ax = plt.subplots()
    format_ms(ax)

    comparison_dir = os.path.join(output_prefix, 'comparison')

    # empty data: nothing to do
    if not data.items():
        return

    width = 0.05
    positions = np.arange(len(data.items()))
    offset = 0

    bars = []
    for network in networks:
        bar = ax.bar(positions + offset, [data[pass_name][network] for pass_name in data.keys()], width)
        offset += width
        bars.append(bar)
    if not bars:
        return

    ax.legend(tuple(bars), tuple(networks))
    ax.set_xticks(positions + width*(len(networks) / 2))
    ax.set_xticklabels([name for (name, _) in data.items()])
    plt.title(title)
    plt.xlabel('Pass Applied')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    outfile = prepare_out_file(comparison_dir, filename)
    plt.savefig(outfile)
    plt.close()


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    devs = config['devices']
    networks = config['networks']

    # read in data, output graphs of most recent data, and output longitudinal graphs
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    last_two_weeks = [entry for entry in all_data
                      if time_difference(most_recent, entry).days < 14]

    try:
        generate_longitudinal_comparisons(all_data, output_dir, 'all_time')
        generate_longitudinal_comparisons(last_two_weeks, output_dir, 'two_weeks')
        for dev in devs:
            generate_relay_opt_comparisons('Individual Passes Applied on {}'.format(dev.upper()),
                                           'pass-comparison-{}.png'.format(dev), most_recent[dev],
                                           networks, output_dir)

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
