import os

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import numpy as np

from validate_config import validate
from common import (invoke_main, write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import format_ms, generate_longitudinal_comparisons


def generate_training_loop_comparison(title, filename, data, epochs, output_prefix=''):
    fig, ax = plt.subplots()
    format_ms(ax)

    comparison_dir = os.path.join(output_prefix, 'comparison')

    # empty data: nothing to do
    if not data.items():
        return

    width = 0.05
    positions = np.arange(len(epochs))
    offset = 0

    bars = []
    for (_, measurements) in data.items():
        # quirk of JSON: integers are not allowed to be keys to dicts, so an int-keyed
        # dict in Python turns into a dict with a _string_ version of the keys in JSON,
        # which then still has string keys when converted back into Python. Go figure
        bar = ax.bar(positions + offset, [measurements[str(count)] for count in epochs], width)
        offset += width
        bars.append(bar)
    if not bars:
        return

    ax.legend(tuple(bars), tuple([name for (name, _) in data.items()]))
    ax.set_xticks(positions + width*(len(data.items()) / 2))
    ax.set_xticklabels(tuple(epochs))
    plt.title(title)
    plt.xlabel('Total Epochs')
    plt.ylabel('Mean Time per Epoch (ms)')
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
    epochs = config['epochs']

    # read in data, output graphs of most recent data, and output longitudinal graphs
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    last_two_weeks = [entry for entry in all_data
                      if time_difference(most_recent, entry).days < 14]

    try:
        generate_longitudinal_comparisons(all_data, output_dir, 'all_time')
        generate_longitudinal_comparisons(last_two_weeks, output_dir, 'two_weeks')
        for dev in devs:
            generate_training_loop_comparison('Training Loop Comparison on {}'.format(dev.upper()),
                                              'training_loop-{}.png'.format(dev),
                                              most_recent[dev], epochs, output_dir)
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        return

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
