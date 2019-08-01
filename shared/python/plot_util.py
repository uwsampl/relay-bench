"""Utility definitions for Matplotlib"""
import datetime
import enum
import itertools
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from common import prepare_out_file, gather_stats, traverse_fields

MULTI_BAR_WIDTH = 0.05
NUM_Y_TICKS = 10
GRID_COLOR = '#e8e8e8'
PLOT_STYLE = 'seaborn-paper'

class PlotType(enum.Enum):
    # Bar plot with a single bar per x tick
    BAR = 0
    # Bar plot with multiple bars per x tick
    MULTI_BAR = 1
    # Standard line graph
    LINE = 2


class PlotScale(enum.Enum):
    LINEAR = 0
    LOG = 1

    def as_plt_arg(self):
        if self == PlotScale.LINEAR:
            return 'linear'
        elif self == PlotScale.LOG:
            return 'log'


class PlotBuilder:
    def __init__(self):
        fig, ax = plt.subplots()
        self.fix = fig
        self.ax = ax

        self.multi_bar_width = MULTI_BAR_WIDTH
        self.num_y_ticks = NUM_Y_TICKS
        self.style = PLOT_STYLE

    def make(self, plot_type, data):
        # TODO(weberlo): it doesn't seem like the `context` call is influencing
        # the style
        with plt.style.context(self.style):
            # plot-type-specific config
            if plot_type == PlotType.BAR:
                y_data = self._make_bar(data)
            elif plot_type == PlotType.MULTI_BAR:
                y_data = self._make_multi_bar(data)
            elif plot_type == PlotType.LINE:
                y_data = self._make_line(data)

            # change default appearance
            plt.rcParams['font.family'] = ['Roboto', 'DejaVu Sans']
            plt.rcParams['figure.titleweight'] = 'bold'
            plt.rcParams['lines.linewidth'] = 2

            if hasattr(self, 'title'):
                plt.title(self.title, fontweight='bold')
            if hasattr(self, 'x_label'):
                plt.xlabel(self.x_label)
            if hasattr(self, 'y_label'):
                plt.ylabel(self.y_label)
            # NOTE: the axis scales and the y ticks need to be set after the data has been plotted
            if hasattr(self, 'x_scale'):
                plt.xscale(self.x_scale.as_plt_arg())
            if hasattr(self, 'y_scale'):
                plt.yscale(self.y_scale.as_plt_arg())
                if self.y_scale == PlotScale.LOG:
                    self.ax.get_xaxis().get_major_formatter().labelOnlyBase = False
            plt.minorticks_off()
            format_ms(self.ax)
            self._set_y_axis_ticks(y_data)
        return self

    def _make_bar(self, data):
        if len(data.items()) == 0:
            return

        x_locs = np.arange(len(data.items()))
        x_labels = list(data.keys())
        y_data = list(data.values())
        plt.bar(x_locs, y_data, zorder=3)
        plt.xticks(x_locs, x_labels)

        # only use a grid on the y axis
        self.ax.xaxis.grid(False)
        self.ax.yaxis.grid(True, color=GRID_COLOR, zorder=0)

        # TODO(weberlo): change the impl so we don't need to tack on a '0' to
        # get a lower bound of 0.
        return y_data + [0]

    def _make_multi_bar(self, data):
        if len(data) == 0:
            return

        bar_types = data.keys()
        tick_labels = list(data.values())[0].keys()
        positions = np.arange(len(tick_labels))
        offset = 0
        bars = []
        for framework_data in data.values():
            bar = self.ax.bar(
                    positions + offset,
                    list(framework_data.values()),
                    MULTI_BAR_WIDTH,
                    zorder=3)
            offset += MULTI_BAR_WIDTH
            bars.append(bar)
        if not bars:
            return
        self.ax.legend(tuple(bars), tuple(bar_types))
        plt.xticks(positions + MULTI_BAR_WIDTH*(len(data) / 2),
                   tuple(tick_labels))

        # flatten nested dictionaries to get raw values
        y_data = list(
                itertools.chain.from_iterable(
                    map(lambda x:
                        list(x.values()), data.values())))
        self._set_y_axis_ticks(y_data)

        # only use a grid on the y axis
        self.ax.xaxis.grid(False)
        self.ax.yaxis.grid(True, color=GRID_COLOR, zorder=0)

        return y_data

    def _make_line(self, data):
        x_data = data['x']
        y_data = data['y']
        if len(x_data) == 0 or len(y_data) == 0:
            return

        if isinstance(x_data[0], dict) and 'label' in x_data[0]:
            assert False
            # create tick labels on the x_axis, if they're specified
            x_locs = list(map(lambda x: x['loc'], x_data))
            x_labels = list(map(lambda x: x['label'], x_data))
            plt.plot(x_locs, data['y'])
            plt.xticks(x_locs, x_labels)
        else:
            # otherwise, plot it normally
            plt.plot(x_data, y_data)

        # rotate dates on the x axis, so they don't overlap
        if isinstance(x_data[0], datetime.datetime):
            plt.gcf().autofmt_xdate()

        self.ax.grid(True, color=GRID_COLOR, zorder=0)

        return y_data

    def save(self, dirname, filename):
        outfile = prepare_out_file(dirname, filename)
        plt.savefig(outfile, dpi=500, bbox_inches='tight')
        plt.close()

    def set_style(style_name):
        self.style = style_name
        return self

    def set_title(self, title):
        self.title = title
        return self

    def set_x_label(self, x_label):
        self.x_label = x_label
        return self

    def set_y_label(self, y_label):
        self.y_label = y_label
        return self

    def set_x_scale(self, scale):
        self.x_scale = scale
        return self

    def set_y_scale(self, scale):
        self.y_scale = scale
        return self

    def _set_y_axis_ticks(self, y_values, num_ticks=NUM_Y_TICKS):
        y_min = min(y_values)
        y_max = max(y_values)
        if not hasattr(self, 'y_scale') or self.y_scale == PlotScale.LINEAR:
            self.ax.set_yticks(np.linspace(y_min, y_max, num=num_ticks))
        elif self.y_scale == PlotScale.LOG:
            start = np.log10(y_min)
            stop = np.log10(y_max)
            self.ax.set_yticks(np.logspace(start, stop, base=10, num=num_ticks))


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

        PlotBuilder() \
            .set_title('({}) over Time'.format(','.join(fields))) \
            .set_x_label('Date of Run') \
            .set_y_label('Time (ms)') \
            .make(PlotType.LINE, {'x': times, 'y': stats}) \
            .save(longitudinal_dir, 'longitudinal-{}.png'.format('-'.join(fields)))
