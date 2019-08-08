"""Utility definitions for Matplotlib"""
import datetime
import enum
import itertools
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties

from common import prepare_out_file, gather_stats, traverse_fields

BAR_LABEL_Y_COEFF = 1.05
BAR_WIDTH = 0.2
MULTI_BAR_Y_TOP_PAD_COEFF = 1.17
NUM_Y_TICKS = 5
GRID_COLOR = '#e8e8e8'
PLOT_STYLE = 'seaborn-paper'
LINEAR_AXIS_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
LOG_AXIS_BASES = [2, 5, 10]

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
        self.bar_width = BAR_WIDTH
        self.num_y_ticks = NUM_Y_TICKS
        self.style = PLOT_STYLE
        self.figsize = plt.rcParams['figure.figsize']

    def make(self, plot_type, data):
        self.plot_type = plot_type

        fig, ax = plt.subplots(figsize=self.figsize)
        self.fig = fig
        self.ax = ax
        # TODO(weberlo): it doesn't seem like the `context` call is influencing
        # the style
        #with plt.style.context(self.style):

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
        self._set_up_y_axis(y_data)
        return self

    def _make_bar(self, data):
        if len(data.items()) == 0:
            return

        x_locs = np.arange(len(data.items()))
        x_labels = list(data.keys())
        y_data = list(data.values())
        # TODO(weberlo): handle unit conversions better
        # seconds to milliseconds
        y_data = np.array(y_data) * 1e3
        bar = plt.bar(x_locs, y_data, zorder=3)
        self._label_bar_val(bar, y_data)
        plt.xticks(x_locs, x_labels)

        return self._post_bar_setup(y_data)

    def _make_multi_bar(self, data):
        if len(data) == 0:
            return

        bar_types = data.keys()
        tick_labels = list(data.values())[0].keys()
        positions = np.arange(len(tick_labels))
        offset = 0
        bars = []
        for framework_data in data.values():
            # TODO(weberlo): handle unit conversions better
            # seconds to milliseconds
            single_y_data = np.array(list(framework_data.values())) * 1e3
            bar = self.ax.bar(
                    positions + offset,
                    single_y_data,
                    self.bar_width,
                    zorder=3)
            offset += self.bar_width
            self._label_bar_val(bar, single_y_data)
            bars.append(bar)
        if not bars:
            return

        font_prop = FontProperties()
        font_prop.set_size('small')
        self.ax.legend(tuple(bars), tuple(bar_types),
                fancybox=False,
                framealpha=1.0,
                facecolor='white',
                edgecolor='black',
                loc='upper center',
                bbox_to_anchor=(0.5, 1.0),
                ncol=len(bars),
                prop=font_prop)

        # center x ticks in the middle of each multi-bar cluster and add labels
        x_tick_positions = positions + self.bar_width*((len(data) - 1) / 2)
        plt.xticks(x_tick_positions, tuple(tick_labels))

        # flatten nested dictionaries to get raw values
        y_data = list(
                itertools.chain.from_iterable(
                    map(lambda x:
                        list(x.values()), data.values())))
        # TODO(weberlo): handle unit conversions better
        # seconds to milliseconds
        y_data = list(np.array(y_data) * 1e3)

        return self._post_bar_setup(y_data)

    def _make_line(self, data):
        x_data = data['x']
        y_data = data['y']
        if len(x_data) == 0 or len(y_data) == 0:
            return
        # TODO(weberlo): handle unit conversions better
        # seconds to milliseconds
        y_data = list(np.array(y_data) * 1e3)

        # plot the line
        plt.plot(x_data, y_data, zorder=3)
        # then emphasize the data points that shape the line
        plt.scatter(x_data, y_data, zorder=3)

        # rotate dates on the x axis, so they don't overlap
        if isinstance(x_data[0], datetime.datetime):
            plt.gcf().autofmt_xdate()

        self.ax.grid(True, color=GRID_COLOR, zorder=0)

        return y_data

    def _post_bar_setup(self, y_data):
        # only use a grid on the y axis
        self.ax.xaxis.grid(False)
        self.ax.yaxis.grid(True, color=GRID_COLOR, zorder=0)
        # disable x-axis ticks (but show labels) by setting the tick length to 0
        self.ax.tick_params(axis='both', which='both',length=0)

        # TODO(weberlo): change the impl so we don't need to tack on values to
        # get a lower bound of 0.
        if hasattr(self, 'y_scale') and self.y_scale == PlotScale.LOG:
            return y_data + [1]
        else:
            return y_data + [0]


    def _label_bar_val(self, bar_container, y_data):
        def _format_val(val):
            # NOTE: This function is full of gross heuristics.

            # Try a format option where we force the number of significant
            # figures and another format option that forces the number of
            # decimal places. We use whichever results in a shorter string.
            force_sig_figs_text = '{:.2}'.format(val)
            force_num_decimal_places_text = '{:.2f}'.format(val)
            if len(force_sig_figs_text) < len(force_num_decimal_places_text):
                text = force_sig_figs_text
            else:
                text = force_num_decimal_places_text

            if 'e+' in text or (val > 1.0 and len(text) > 4):
                # If scientific notation with a positive exponent is required
                # to represent it or too many decimals are required, just show
                # it as an int.
                text = '{}'.format(int(val))
            return text

        for bar, val in zip(bar_container.get_children(), y_data):
            self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * BAR_LABEL_Y_COEFF,
                         _format_val(val),
                         ha='center', va='bottom')

    def _set_up_y_axis(self, y_data):
        def _choose_linear_step(y_min, y_max):
            best_step = LINEAR_AXIS_STEPS[0]
            best_step_start = None
            best_step_stop = None
            min_diff = float('inf')
            for step in LINEAR_AXIS_STEPS:
                step_start = int(y_min / step) * step
                step_stop = int((y_max / step) + 1) * step
                diff = abs(((step_stop - step_start) / step) - self.num_y_ticks)
                if diff <= min_diff:
                    best_step = step
                    best_step_start = step_start
                    best_step_stop = step_stop
                    min_diff = diff
            return best_step, best_step_start, best_step_stop

        def _choose_log_base(y_min, y_max):
            best_base = LOG_AXIS_BASES[0]
            best_base_start = None
            best_base_stop = None
            min_diff = float('inf')
            for base in LOG_AXIS_BASES:
                base_start = int(np.log(y_min) / np.log(base))
                base_stop = int((np.log(y_max) / np.log(base)) + 1)
                diff = abs((base_stop - base_start) - self.num_y_ticks)
                if diff <= min_diff:
                    best_base = base
                    min_diff = diff
                    best_base_start = base_start
                    best_base_stop = base_stop
            return best_base, best_base_start, best_base_stop

        y_data = list(y_data)

        if not hasattr(self, 'y_scale') or self.y_scale == PlotScale.LINEAR:
            y_scale = PlotScale.LINEAR
        else:
            y_scale = self.y_scale

        if self.plot_type == PlotType.BAR or self.plot_type == PlotType.MULTI_BAR:
            # we want to have the bottom of the bars as low as possible on bar plots
            if y_scale == PlotScale.LOG:
                # can't use 0 on log-scale plots
                y_data += [1]
            else:
                y_data += [0]

        y_min = min(y_data)
        y_max = max(y_data)
        if y_scale == PlotScale.LINEAR:
            step, start, stop = _choose_linear_step(y_min, y_max)
            self.ax.set_yticks([i for i in range(start, stop+1, step)])
        elif y_scale == PlotScale.LOG:
            base, start, stop = _choose_log_base(y_min, y_max)
            self.ax.set_yticks([base**i for i in range(start, stop+1)])

        if self.plot_type == PlotType.MULTI_BAR:
            # add padding on the top to make room for bar labels
            if y_scale == PlotScale.LOG:
                y_max = np.power(base, (np.log(y_max) / np.log(base)) * MULTI_BAR_Y_TOP_PAD_COEFF)
            else:
                y_max *= MULTI_BAR_Y_TOP_PAD_COEFF
            self.ax.set_ylim([y_min, y_max])


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

    def set_bar_width(self, bar_width):
        self.bar_width = bar_width
        return self

    def set_figsize(self, figsize):
        self.figsize = figsize
        return self


def format_ms(ax):
    def milliseconds(value, tick_position):
        return '{:3.1f}'.format(value)
    formatter = FuncFormatter(milliseconds)
    ax.yaxis.set_major_formatter(formatter)


def generate_longitudinal_comparisons(sorted_data, output_dir,
                                      subdir_name='longitudinal'):
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
    longitudinal_dir = os.path.join(output_dir, subdir_name)

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
