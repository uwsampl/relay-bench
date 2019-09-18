"""Utility definitions for Matplotlib"""
import datetime
import enum
import functools
import itertools
import logging
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

NUM_SIG_FIGS = 3

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
    # longitudinal graph
    LONGITUDINAL = 2

    def is_bar_variant(self):
        return self in {PlotType.BAR, PlotType.MULTI_BAR}


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
        self.unit_type = UNIT_TYPE
        self.sig_figs = NUM_SIG_FIGS

    def make(self, plot_type, data):
        if len(data) == 0:
            raise RuntimeError('no data to plot')

        self.plot_type = plot_type

        sns.set(style='darkgrid')
        sns.set_context('paper')

        if hasattr(self, 'font_scale'):
            sns.set(font_scale=self.font_scale)

        plt.figure()

        # plot-type-specific config
        if plot_type == PlotType.BAR:
            y_data = BarPlotter(self).make(data)
        elif plot_type == PlotType.MULTI_BAR:
            y_data = CatPlotter(self).make(data)
        elif plot_type == PlotType.LONGITUDINAL:
            y_data = LongitudinalPlotter(self).make(data)
        else:
            raise RuntimeError(f'unknown plot type "{plot_type}"')

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

        return self

    def save(self, dirname, filename):
        outfile = prepare_out_file(dirname, filename)
        plt.savefig(outfile, dpi=500, bbox_inches='tight')
        plt.close()

    def filter_y_val(self, val):
        if not _is_valid_num(val):
            logging.warning(f'found invalid value "{val}" in data. Plot results may suffer')
            return None

        if self.unit_type == UnitType.SECONDS:
            return val * 1e3
        elif self.unit_type in (UnitType.MILLISECONDS, UnitType.COMPARATIVE):
            return val
        else:
            raise RuntimeError(f'unhandled unit type "{self.unit_type}"')

    def post_bar_setup(self):
        # only use a grid on the y axis
        self.ax().xaxis.grid(False)
        # disable x-axis ticks (but show labels) by setting the tick length to 0
        self.ax().tick_params(axis='both', which='both',length=0)

    def label_bar_val(self, bar_container, all_data_mean):
        def _format_val(val):
            sig_figs = self.sig_figs
            if val < 1.0:
                sig_figs -= 1
            # g = use significant figures
            sig_fig_format = '{{:#.{}g}}'.format(sig_figs)
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


class BarPlotter:
    def __init__(self, builder):
        self.builder = builder

    def make(self, data):
        self.process_data(data)
        df = self.to_dataframe(data)
        all_y_data = list(df[data['meta'][1]])
        all_data_mean = np.mean(all_y_data)

        metadata = data['meta']
        kwargs = {
            'x': metadata[0],
            'y': metadata[1],
            'data': df,
        }

        if hasattr(self.builder, 'aspect_ratio'):
            kwargs['aspect'] = self.builder.aspect_ratio
        if hasattr(self.builder, 'figure_height'):
            kwargs['height'] = self.builder.figure_height
        if hasattr(self.builder, 'bar_colors'):
            kwargs['palette'] = self.builder.bar_colors
        g = sns.barplot(**kwargs)

        bar_containers = g.axes.containers
        for container in bar_containers:
            self.builder.label_bar_val(container, all_data_mean)

        self.builder.post_bar_setup()
        return all_y_data

    def process_data(self, data):
        for path, val in _traverse_dict(data['raw']):
            new_val = self.builder.filter_y_val(val)
            if new_val is None:
                del data[path[0]]
            else:
                data[path[0]] = new_val

    def to_dataframe(self, data):
        return pd.DataFrame({
            data['meta'][0]: list(data['raw'].keys()),
            data['meta'][1]: list(data['raw'].values()),
        })


class CatPlotter:
    def __init__(self, builder):
        self.builder = builder

    def make(self, data):
        self.process_data(data)
        df = self.to_dataframe(data)

        all_y_data = list(df[data['meta'][2]])
        all_data_mean = np.mean(all_y_data)

        # TODO: Could remove `kind='bar'` to switch to scatter, use
        # `g.axes[0][0].get_collections()`, spread the points in each bucket,
        # then use  `g.axes[0][0].set_collections()` to update it.
        #for collection in g.axes[0][0].collections:
        #    self._label_scatter_val(collection, all_data_mean)
        metadata = data['meta']
        kwargs = {
            'x': metadata[1],
            'y': metadata[2],
            'hue': metadata[0],
            'data': df,
            'kind': 'bar',
            'legend_out': True
        }
        if hasattr(self.builder, 'aspect_ratio'):
            kwargs['aspect'] = self.builder.aspect_ratio
        if hasattr(self.builder, 'figure_height'):
            kwargs['height'] = self.builder.figure_height
        if hasattr(self.builder, 'bar_colors'):
            kwargs['palette'] = self.builder.bar_colors
        g = sns.catplot(**kwargs)
        bar_containers = g.axes[0][0].containers
        for container in bar_containers:
            self.builder.label_bar_val(container, all_data_mean)

        self.builder.post_bar_setup()
        return all_y_data

    def process_data(self, data):
        raw_data = data['raw']
        for path, val in _traverse_dict(raw_data):
            new_val = self.builder.filter_y_val(val)
            if new_val is None:
                del raw_data[path[0]][path[1]]
            else:
                raw_data[path[0]][path[1]] = new_val

    def to_dataframe(self, data):
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


class LongitudinalPlotter:
    def __init__(self, builder):
        self.builder = builder

    def make(self, data):
        self.process_data(data)
        x_data = data['raw']['x']
        y_data = data['raw']['y']
        if len(x_data) == 0 or len(y_data) == 0:
            raise RuntimeError('no data to plot')
        metadata = data['meta']
        df = self.to_dataframe(data)

        # plot the line
        sns.lineplot(x=metadata[0], y=metadata[1], data=df)
        # then emphasize the data points that shape the line
        plt.scatter(x_data, y_data, zorder=3)

        # rotate dates on the x axis, so they don't overlap
        if isinstance(x_data[0], datetime.datetime):
            plt.gcf().autofmt_xdate()

        return y_data

    def process_data(self, data):
        raw_data = data['raw']
        to_remove = []
        for idx, (label, val) in enumerate(zip(raw_data['x'], raw_data['y'])):
            new_val = self.builder.filter_y_val(val)
            if new_val is None:
                to_remove.append(idx)
            else:
                raw_data['y'][idx] = new_val
        # reverse removal since we're deleting while iterating
        for idx in reversed(to_remove):
            del raw_data['x'][idx]
            del raw_data['y'][idx]

    def to_dataframe(self, data):
        return pd.DataFrame({
            data['meta'][0]: data['raw']['x'],
            data['meta'][1]: data['raw']['y'],
        })


def generate_longitudinal_comparisons(sorted_data, output_dir,
                                      subdir_name='longitudinal',
                                      stat_name='Time (ms)'):
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

        data = {
            'raw': {'x': times, 'y': stats},
            'meta': ['Date of Run', stat_name]
        }
        PlotBuilder() \
            .set_title('({}) over Time'.format(','.join(fields))) \
            .set_x_label(data['meta'][0]) \
            .set_y_label(data['meta'][1]) \
            .make(PlotType.LONGITUDINAL, data) \
            .save(longitudinal_dir, 'longitudinal-{}.png'.format('-'.join(fields)))


def _is_valid_num(val):
    return val is not None and not np.isnan(val)


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
