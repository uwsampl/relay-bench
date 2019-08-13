"""Utility definitions for Matplotlib"""
import datetime
import enum
import functools
import itertools
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.ticker import FuncFormatter
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pandas as pd

from common import prepare_out_file, gather_stats, traverse_fields

MIN_NUM_Y_TICKS = 2
TARGET_NUM_Y_TICKS = 5
MAX_NUM_Y_TICKS = 10
LINEAR_AXIS_STEPS = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
LOG_AXIS_BASES = [2, 10]
NUM_SIG_FIGS = 4

class UnitType(enum.Enum):
    SECONDS = 0
    MILLISECONDS = 1
    # Speedup or Slowdown
    COMPARATIVE = 2


UNIT_TYPE = UnitType.SECONDS

class PlotType(enum.Enum):
    # Bar plot with a single bar per x tick
    BAR = 0
    # Bar plot with multiple bars per x tick
    MULTI_BAR = 1
    # Standard line graph
    LINE = 2

    def is_bar_variant(self):
        return self in {PlotType.BAR, PlotType.MULTI_BAR}


Y_TOP_PAD_COEFF = {
    PlotType.BAR: 1.1,
    PlotType.MULTI_BAR: 1.07,
    PlotType.LINE: 1.0
}

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
        self.figsize = plt.rcParams['figure.figsize']
        self.unit_type = UNIT_TYPE
        self.sig_figs = NUM_SIG_FIGS

    def make(self, plot_type, data):
        self.plot_type = plot_type

        sns.set(style='darkgrid')
        sns.set_context('paper')

        if hasattr(self, 'font_scale'):
            sns.set(font_scale=self.font_scale)

        plt.figure(figsize=self.figsize)

        # plot-type-specific config
        if plot_type == PlotType.BAR:
            y_data = self._make_bar(data)
        elif plot_type == PlotType.MULTI_BAR:
            y_data = self._make_multi_bar(data)
        elif plot_type == PlotType.LINE:
            y_data = self._make_line(data)

        if hasattr(self, 'title'):
            plt.title(self.title)
        if hasattr(self, 'x_label'):
            plt.xlabel(self.x_label)
        else:
            plt.xlabel('')
        if hasattr(self, 'y_label'):
            plt.ylabel(self.y_label)
        else:
            plt.ylabel('')
        # NOTE: the axis scales and the y ticks need to be set after the data has been plotted
        if hasattr(self, 'x_scale'):
            plt.xscale(self.x_scale.as_plt_arg())
        if hasattr(self, 'y_scale'):
            plt.yscale(self.y_scale.as_plt_arg())
        plt.minorticks_off()
        self._set_up_y_axis(y_data)

        return self

    def save(self, dirname, filename):
        outfile = prepare_out_file(dirname, filename)
        plt.savefig(outfile, dpi=500, bbox_inches='tight')
        plt.close()

    def _make_bar(self, data):
        if len(data.items()) == 0:
            return

        x_locs = np.arange(len(data.items()))
        x_labels = list(data.keys())
        y_data = self._process_y_data(list(data.values()))
        bar = plt.bar(x_locs, y_data, zorder=3)
        self._label_bar_val(bar, y_data, np.mean(y_data))
        plt.xticks(x_locs, x_labels)

        return self._post_bar_setup(y_data)

    def _make_multi_bar(self, data):
        if len(data) == 0:
            return

        raw_data = data['raw']
        metadata = data['meta']

        # flatten nested dictionaries to get raw values
        all_y_data = list(
                itertools.chain.from_iterable(
                    map(lambda x:
                        list(x.values()), raw_data.values())))
        all_y_data = self._process_y_data(all_y_data)
        all_data_mean = np.mean(list(filter(_is_valid_num, all_y_data)))

        df = to_dataframe(data)

        # TODO: Could remove `kind='bar'` to switch to scatter, use
        # `g.axes[0][0].get_collections()`, spread the points in each bucket,
        # then use  `g.axes[0][0].set_collections()` to update it.
        #for collection in g.axes[0][0].collections:
        #    self._label_scatter_val(collection, all_data_mean)
        kwargs = {
            'x': metadata[1],
            'y': metadata[2],
            'hue': metadata[0],
            'data': df,
            'kind': 'bar',
            'legend_out': True
        }
        if hasattr(self, 'aspect_ratio'):
            kwargs['aspect'] = self.aspect_ratio
        if hasattr(self, 'figure_height'):
            kwargs['height'] = self.figure_height
        if hasattr(self, 'bar_colors'):
            kwargs['palette'] = self.bar_colors
        g = sns.catplot(**kwargs)
        bar_containers = g.axes[0][0].containers
        for container in bar_containers:
            self._label_bar_val(container, all_data_mean)

        return self._post_bar_setup(all_y_data)

    def _make_line(self, data):
        x_data = data['x']
        y_data = data['y']
        if len(x_data) == 0 or len(y_data) == 0:
            return
        y_data = self._process_y_data(y_data)

        # plot the line
        plt.plot(x_data, y_data, zorder=3)
        # then emphasize the data points that shape the line
        plt.scatter(x_data, y_data, zorder=3)

        # rotate dates on the x axis, so they don't overlap
        if isinstance(x_data[0], datetime.datetime):
            plt.gcf().autofmt_xdate()

        return y_data

    def _process_y_data(self, y_data):
        # TODO(weberlo): this is a garbage way to deal with NaNs
        # replace NaNs/Nones with the minimum y value
        min_y = min(filter(_is_valid_num, y_data))
        without_nans = np.array([min_y if not _is_valid_num(y) else y for y in y_data])
        if self.unit_type == UnitType.SECONDS:
            # TODO(weberlo): handle unit conversions better
            # seconds to milliseconds
            return list(without_nans * 1e3)
        else:
            return list(without_nans)

    def _post_bar_setup(self, y_data):
        # only use a grid on the y axis
        self.ax().xaxis.grid(False)
        # disable x-axis ticks (but show labels) by setting the tick length to 0
        self.ax().tick_params(axis='both', which='both',length=0)

        # TODO(weberlo): change the impl so we don't need to tack on values to
        # get a lower bound of 0.
        if hasattr(self, 'y_scale') and self.y_scale == PlotScale.LOG:
            return y_data + [1]
        else:
            return y_data + [0]

    def _label_bar_val(self, bar_container, all_data_mean):
        def _format_val(val):
            # g = use significant figures
            sig_fig_format = '{{:#.{}g}}'.format(self.sig_figs)
            text = sig_fig_format.format(val)

            if 'e+' in text and len(text) > 6:
                # If scientific notation with a large positive exponent is required
                # to represent it, use fewer sig figs
                text = '{:#.1g}'.format(val)
            return text

        for bar in bar_container.get_children():
            # The height of the bar is in the data space, so it's equivalent to
            # the value that was used to plot it.
            bar_val = bar.get_height()
            # TODO(weberlo): 0.0 should be considered valid
            if _is_valid_num(bar_val) and bar_val != 0.0:
                if self.y_scale == PlotScale.LINEAR:
                    label_height = bar_val + all_data_mean * 0.03
                else:
                    label_height = bar_val * 1.05
                self.ax().text(bar.get_x() + bar.get_width()/2, label_height,
                             _format_val(bar_val),
                             ha='center', va='bottom',
                             size='x-small')

    def _label_scatter_val(self, collection, all_data_mean):
        def _format_val(val):
            # g = use significant figures
            sig_fig_format = '{{:#.{}g}}'.format(self.sig_figs)
            text = sig_fig_format.format(val)

            if 'e+' in text and len(text) > 6:
                # If scientific notation with a large positive exponent is required
                # to represent it, use fewer sig figs
                text = '{:#.1g}'.format(val)
            return text

        for point in collection.get_offsets():
            # The height of the point is in the data space, so it's equivalent to
            # the value that was used to plot it.
            x = point[0]
            y = point[1]
            # TODO(weberlo): 0.0 should be considered valid
            if _is_valid_num(y) and y != 0.0:
                if self.y_scale == PlotScale.LINEAR:
                    label_height = y + all_data_mean * 0.03
                else:
                    label_height = y * 1.05
                self.ax().text(x, label_height,
                             _format_val(y),
                             ha='center', va='bottom',
                             size='x-small')

    def _set_up_y_axis(self, y_data):
        # TODO(weberlo): Refactor `_choose_linear_step` and `_choose_log_base`.
        def _choose_linear_step(y_min, y_max):
            best_step = None
            best_step_start = None
            best_step_stop = None
            min_diff = float('inf')
            for step in LINEAR_AXIS_STEPS:
                step_start = int(y_min / step) * step
                step_stop = int((y_max / step) + 1) * step
                num_y_ticks = (step_stop - step_start) / step
                diff = abs(num_y_ticks - TARGET_NUM_Y_TICKS)
                if diff <= min_diff and MIN_NUM_Y_TICKS < num_y_ticks < MAX_NUM_Y_TICKS:
                    best_step = step
                    best_step_start = step_start
                    best_step_stop = step_stop
                    min_diff = diff
            if best_step is None:
                raise RuntimeError('no suitable linear step could be found')
            return best_step, best_step_start, best_step_stop

        def _choose_log_base(y_min, y_max):
            best_base = None
            best_base_start = None
            best_base_stop = None
            min_diff = float('inf')
            for base in LOG_AXIS_BASES:
                base_start = int(np.log(y_min) / np.log(base))
                base_stop = int((np.log(y_max) / np.log(base)) + 1)
                num_y_ticks = base_stop - base_start
                diff = abs(num_y_ticks - TARGET_NUM_Y_TICKS)
                if diff <= min_diff and MIN_NUM_Y_TICKS < num_y_ticks < MAX_NUM_Y_TICKS:
                    best_base = base
                    min_diff = diff
                    best_base_start = base_start
                    best_base_stop = base_stop
            if best_base is None:
                raise RuntimeError('no suitable log base could be found')
            return best_base, best_base_start, best_base_stop

        def _format_sub_one(value, tick_position):
            return '{:3.1f}'.format(value)

        def _format_int(value, tick_position):
            return '{}'.format(int(value))

        y_data = list(y_data)

        if not hasattr(self, 'y_scale') or self.y_scale == PlotScale.LINEAR:
            y_scale = PlotScale.LINEAR
        else:
            y_scale = self.y_scale

        if self.plot_type.is_bar_variant():
            # we want to have the bottom of the bars as low as possible on bar plots
            if y_scale == PlotScale.LOG:
                # can't use 0 on log-scale plots
                y_data += [1]
            else:
                y_data += [0]

        y_min = min(y_data)
        y_max = max(y_data)
        #formatter = _format_sub_one
        if y_scale == PlotScale.LINEAR:
            step, start, stop = _choose_linear_step(y_min, y_max)
            #if step is not None:
            #    if step >= 1.0:
            #        formatter = _format_int
            #    self.ax().set_yticks([i for i in np.arange(start, stop+step, step)])
        elif y_scale == PlotScale.LOG:
            base, start, stop = _choose_log_base(y_min, y_max)
            #formatter = _format_int
            #if base is not None:
            #    self.ax().set_yticks([int(base**i) for i in range(start, stop+1)])

        if self.plot_type.is_bar_variant():
            y_top_pad_coeff = Y_TOP_PAD_COEFF[self.plot_type]
            # add padding on the top to make room for bar labels
            if y_scale == PlotScale.LOG:
                y_max = np.power(base, (np.log(y_max) / np.log(base)) * y_top_pad_coeff)
            else:
                y_max *= y_top_pad_coeff
            self.ax().set_ylim([y_min, y_max])

        #self.ax().yaxis.set_major_formatter(FuncFormatter(formatter))

    def fig(self):
        return plt.gcf()

    def ax(self):
        return plt.gca()

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

    def set_aspect_ratio(self, aspect_ratio):
        self.aspect_ratio = aspect_ratio
        return self

    def set_figure_height(self, figure_height):
        self.figure_height = figure_height
        return self

    def set_figsize(self, figsize):
        self.figsize = figsize
        return self

    def set_sig_figs(self, sig_figs):
        self.sig_figs = sig_figs
        return self

    def set_unit_type(self, unit_type):
        self.unit_type = unit_type
        return self

    def set_bar_colors(self, bar_colors):
        self.bar_colors = bar_colors
        return self

    def set_font_scale(self, font_scale):
        self.font_scale = font_scale
        return self


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


def _is_valid_num(val):
    return val is not None and not np.isnan(val)


def to_dataframe(data):
    outer_ordering = list(list(data['raw'].values())[0].keys())
    def _cmp_func(a, b):
        return outer_ordering.index(a[0]) - outer_ordering.index(b[0])

    def _unzip(lst):
        a_lst = []
        b_lst = []
        c_lst = []
        for (a, b, c) in lst:
            a_lst.append(a)
            b_lst.append(b)
            c_lst.append(c)
        return (a_lst, b_lst, c_lst)

    raw_data = data['raw']
    metadata = data['meta']
    df_data = []
    for path, val in _traverse_dict(raw_data):
        df_data.append((path[1], path[0], val))
    df_data.sort(key=functools.cmp_to_key(_cmp_func))
    a_lst, b_lst, c_lst = _unzip(df_data)
    return pd.DataFrame({
        metadata[1]: a_lst,
        metadata[0]: b_lst,
        metadata[2]: c_lst
    })


def _traverse_dict(dic, path=None):
    if path is None:
        path = []
    for (key, val) in dic.items():
        local_path = list(path)
        local_path.append(key)
        if isinstance(val, dict):
            for j in _traverse_dict(val, local_path):
                yield j
        else:
            yield local_path, val


