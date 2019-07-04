import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import argparse
import datetime
import json
import numpy as np
import os

def prepare_out_file(output_prefix, filename):
    full_name = os.path.join(output_prefix, filename)
    if not os.path.exists(os.path.dirname(full_name)):
        os.makedirs(os.path.dirname(full_name))
    return full_name


def format_ms(ax):
    def milliseconds(value, tick_position):
        return '{:3.1f}'.format(value*1e3)
    formatter = FuncFormatter(milliseconds)
    ax.yaxis.set_major_formatter(formatter)


def generate_relay_cnn_opt_comparisons(title, filename, data, output_prefix=''):
    fig, ax = plt.subplots()
    format_ms(ax)

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


def generate_cnn_comparisons(title, filename, data, output_prefix=''):
    fig, ax = plt.subplots()
    format_ms(ax)

    # empty data: nothing to do
    if not data.items():
        return

    networks = list(list(data.items())[0][1].keys())

    width = 0.05
    positions = np.arange(len(networks))
    offset = 0

    bars = []
    for (_, measurements) in data.items():
        bar = ax.bar(positions + offset, [measurements[network] for network in networks], width)
        offset += width
        bars.append(bar)
    if not bars:
        return

    ax.legend(tuple(bars), tuple([name for (name, _) in data.items()]))
    ax.set_xticks(positions + width*(len(data.items()) / 2))
    ax.set_xticklabels(tuple(networks))
    plt.title(title)
    plt.xlabel('Network')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    outfile = prepare_out_file(output_prefix, filename)
    plt.savefig(outfile)


def generate_char_rnn_comparison(title, filename, data, output_prefix=''):
    fig, ax = plt.subplots()
    format_ms(ax)

    means = [measurements['char-rnn'] for (_, measurements) in data.items()]
    if not means:
        return

    settings = np.arange(len(data.items()))
    plt.bar(settings, means)
    plt.xticks(settings, [name for (name, _) in data.items()])
    plt.title(title)
    plt.xlabel('Framework')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    outfile = prepare_out_file(output_prefix, filename)
    plt.savefig(outfile)


def generate_tree_lstm_comparison(title, filename, data, output_prefix=''):
    fig, ax = plt.subplots()
    format_ms(ax)

    means = [measurements['treelstm'] for (_, measurements) in data.items()]
    if not means:
        return

    settings = np.arange(len(data.items()))
    plt.bar(settings, means)
    plt.xticks(settings, [name for (name, _) in data.items()])
    plt.title(title)
    plt.xlabel('Framework')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    outfile = prepare_out_file(output_prefix, filename)
    plt.savefig(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get hidden')
    parser.add_argument("--data-dir", type=str, default='')
    parser.add_argument("--output-dir", type=str, default='')
    args = parser.parse_args()

    graph_settings = {
        'opt-cpu': ('Relay CNN Opt Level on CPU', 'relay-cnn-cpu.png',
                    generate_relay_cnn_opt_comparisons),
        'opt-gpu': ('Relay CNN Opt Level on GPU', 'relay-cnn-gpu.png',
                    generate_relay_cnn_opt_comparisons),
        'cnn-cpu': ('CNN Comparison on CPU', 'cnns-cpu.png',
                    generate_cnn_comparisons),
        'cnn-gpu': ('CNN Comparison on GPU', 'cnns-gpu.png',
                    generate_cnn_comparisons),
        'char-rnn': ('Char RNN Comparison on CPU', 'char-rnns-cpu.png',
                     generate_char_rnn_comparison),
        'treelstm': ('TreeLSTM Comparison on CPU', 'tree-lstm-cpu.png',
                     generate_tree_lstm_comparison)
    }

    # crawl all data files, identify most recent
    for _, _, files in os.walk(args.data_dir):
        most_recent_data = None
        most_recent_time = None

        for name in files:
            with open(os.path.join(args.data_dir, name)) as json_file:
                data = json.load(json_file)
                time = datetime.datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S.%f')

                if most_recent_time is None or time > most_recent_time:
                    most_recent_data = data
                    most_recent_time = time

        for (benchmark, (title, filename, generator)) in graph_settings.items():
            generator(title, filename, most_recent_data[benchmark], args.output_dir)
