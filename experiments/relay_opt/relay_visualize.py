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

def generate_relay_opt_comparisons(title, filename, data, output_prefix=''):
    fig, ax = plt.subplots()
    format_ms(ax)

    comparison_dir = os.path.join(output_prefix, 'comparison')

    # empty data: nothing to do
    if not data.items():
        return

    levels = list(data.keys())
    networks = list(list(data.items())[0][1].keys())

    width = 0.05
    positions = np.arange(len(data.items()))
    offset = 0

    bars = []
    for network in networks:
        bar = ax.bar(positions + offset, [data[level][network] for level in levels], width)
        offset += width
        bars.append(bar)
    if not bars:
        return

    ax.legend(tuple(bars), tuple(networks))
    ax.set_xticks(positions + width*(len(networks) / 2))
    ax.set_xticklabels([name for (name, _) in data.items()])
    plt.title(title)
    plt.xlabel('Opt Level')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    outfile = prepare_out_file(output_prefix, filename)
    plt.savefig(outfile)
    plt.close()


def generate_dumb_longitudinal_comparisons(sorted_data, dev, output_prefix=''):
    if not sorted_data:
        return

    longitudinal_dir = os.path.join(output_prefix, 'longitudinal')

    times = [parse_timestamp(entry) for entry in sorted_data]
    most_recent = sorted_data[-1]
    for (benchmark, measurements) in most_recent.items():
        for (setting, network_times) in measurements.items():
            for (network, _) in network_times.items():
                stats = [entry[benchmark][setting][network] for entry in sorted_data]

                fig, ax = plt.subplots()
                format_ms(ax)
                plt.plot(times, stats)
                plt.title('{}: {} on {} over Time on {}'.format(benchmark, setting, network, dev))
                filename = 'longitudinal-{}-{}-{}-{}.png'.format(benchmark, setting, network, dev)
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

    for dev in devs:
        key = 'opt-{}'.format(dev)
        try:
            generate_relay_opt_comparisons('Relay CNN Opt Level on {}'.format(dev.upper()),
                                           'relay-cnn-{}.png'.format(dev), most_recent[key], output_dir)
            generate_longitudinal_comparisons(list(map(lambda d: d[key], all_data)), output_dir)
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
