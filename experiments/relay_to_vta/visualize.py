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

SIM_TARGETS = {'sim', 'tsim'}
PHYS_TARGETS = {'pynq'}

def generate_longitudinal_comparisons(sorted_data, output_dir):
    longitudinal_dir = os.path.join(output_dir, 'longitudinal')

    most_recent = sorted_data[-1]
    for target in most_recent.keys() & SIM_TARGETS:
        times = [parse_timestamp(entry) for entry in sorted_data if target in entry]
        for stat in most_recent[target].keys():
            stats = [entry[target][stat] for entry in sorted_data if target in entry]
            fig, ax = plt.subplots()
            plt.plot(times, stats)
            plt.title('{} on {} over Time'.format(stat, target))
            filename = 'longitudinal-{}-{}.png'.format(target, stat)
            plt.xlabel('Date of Run')
            plt.ylabel(stat)
            plt.yscale('log')
            plt.gcf().autofmt_xdate()
            outfile = prepare_out_file(longitudinal_dir, filename)
            plt.savefig(outfile)
            plt.close()

    for target in most_recent.keys() & PHYS_TARGETS:
        times = [parse_timestamp(entry) for entry in sorted_data if target in entry]
        means = [entry[target]['mean'] for entry in sorted_data if target in entry]
        std_devs = [entry[target]['std_dev'] for entry in sorted_data if target in entry]
        fig, ax = plt.subplots()
        plt.errorbar(times, means, yerr=std_devs)
        plt.title('Mean Inference Time on {} over Time'.format(target))
        filename = 'longitudinal-{}-inference-time.png'.format(target)
        plt.xlabel('Date of Run')
        plt.ylabel('Inference Time (seconds)')
        plt.gcf().autofmt_xdate()
        outfile = prepare_out_file(longitudinal_dir, filename)
        plt.savefig(outfile)
        plt.close()


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    # read in data, output graphs of most recent data, and output longitudinal graphs
    all_data = sort_data(data_dir)
    generate_longitudinal_comparisons(all_data, output_dir)

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.config_dir, args.output_dir)
