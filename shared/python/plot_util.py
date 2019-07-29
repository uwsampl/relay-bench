"""Utility definitions for Matplotlib"""
import datetime
import enum
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from common import prepare_out_file

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
    _format_ms(ax)
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


def _format_ms(ax):
    def milliseconds(value, tick_position):
        return '{:3.1f}'.format(value*1e3)
    formatter = FuncFormatter(milliseconds)
    ax.yaxis.set_major_formatter(formatter)
