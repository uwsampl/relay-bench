"""Utility definitions for Matplotlib"""
import datetime
import enum
import itertools
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from common import prepare_out_file, gather_stats, traverse_fields

NUM_Y_TICKS = 10
GRID_COLOR = '#e8e8e8'

class PlotType(enum.Enum):
    BAR = 0
    LINE = 1


def make_plot(plot_type, title, x_label, y_label,
              x, y,
              dirname, filename,
              x_tick_labels=None, num_y_ticks=NUM_Y_TICKS, log_scale=False):
    if len(x) == 0 or len(y) == 0:
        return

    fig, ax = plt.subplots()

    # plot-type-specific setup
    if plot_type == PlotType.BAR:
        plt.bar(x, y)
        min_y = 0
    elif plot_type == PlotType.LINE:
        plt.plot(x, y)
        min_y = min(y)
        ax.grid(color=GRID_COLOR, zorder=0)

    # change default appearance
    plt.rcParams['font.family'] = ['Roboto', 'DejaVu Sans']
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['lines.linewidth'] = 2

    plt.title(title, fontweight='bold')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if log_scale:
        plt.yscale('log')
    format_ms(ax)
    # rotate dates on the x axis, so they don't overlap
    if isinstance(x[0], datetime.datetime):
        plt.gcf().autofmt_xdate()
    # create named ticks on the x_axis, if they're specified
    if x_tick_labels:
        plt.xticks(x, x_tick_labels)
    # add evenly-spaced ticks on the y-axis
    max_y = max(y)
    diff = max_y - min_y
    ax.set_yticks([min_y + i * (diff / num_y_ticks) for i in range(num_y_ticks + 1)])

    outfile = prepare_out_file(dirname, filename)
    plt.savefig(outfile, dpi=500, bbox_inches='tight')
    plt.close()


def format_ms(ax):
    def milliseconds(value, tick_position):
        return '{:3.1f}'.format(value*1e3)
    formatter = FuncFormatter(milliseconds)
    ax.yaxis.set_major_formatter(formatter)


def generate_longitudinal_comparisons(sorted_data, output_dir):
    '''
    Generic longitudinal graph generator. Given a list of JSON
    objects sorted by timestamp, generates a
    longitudinal graph for every combination of the
    entries' fields (based on traversing the most recent entry) and
    writes it to output_dir/longitudinal.
    '''
    if not sorted_data:
        return

    field_values = traverse_fields(sorted_data[-1])
    longitudinal_dir = os.path.join(output_dir, 'longitudinal')

    for fields in itertools.product(*field_values):
        stats, times = gather_stats(sorted_data, fields)
        if not stats:
            continue
        title = '({}) over Time'.format(','.join(fields))
        filename = 'longitudinal-{}.png'.format('-'.join(fields))
        make_plot(PlotType.LINE, title,
                  'Date of Run', 'Time (ms)',
                  times, stats,
                  longitudinal_dir, filename,
                  num_y_ticks=7)
