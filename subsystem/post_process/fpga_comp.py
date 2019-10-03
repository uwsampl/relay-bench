import itertools
import os
from collections import OrderedDict

from plot_util import PlotBuilder, PlotScale, PlotType, UnitType
from common import invoke_main, traverse_fields

DICT_DATA = {
    'raw': OrderedDict([
        ('No FPGA', OrderedDict([
            ('ResNet-18', 307.093),
            ('ResNet-34', 568.611),
            ('ResNet-50', 715.668),
            ('DCGAN', 329.37),
            ('TreeLSTM', 131.142)
            ])),
        ('Single-Batch', OrderedDict([
            ('ResNet-18', 64.894),
            ('ResNet-34', 96.912),
            ('ResNet-50', 188.193),
            ('DCGAN', 29.27),
            ('TreeLSTM', 52.867)
            ])),
        ('Multi-Batch', OrderedDict([
            ('ResNet-18', 61.326),
            ('ResNet-34', 84.754),
            ('ResNet-50', 165.056),
            ('DCGAN', 25.217),
            ('TreeLSTM', 47.933)
            ]))
        ]),
    'meta': ['Hardware Design', 'Network', 'Time']
}

def make_plot(title, data, output_dir, filename):
    PlotBuilder().set_title(title) \
                 .set_y_label('Mean Inference Time (ms)') \
                 .set_y_scale(PlotScale.LINEAR) \
                 .set_aspect_ratio(1.7) \
                 .set_figure_height(3) \
                 .set_font_scale(0.7) \
                 .set_sig_figs(3) \
                 .set_bar_colors(['C3', 'C4', 'C6']) \
                 .set_unit_type(UnitType.MILLISECONDS) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)


def main(data_dir, output_dir):
    make_plot('FPGA', DICT_DATA, output_dir, 'fpga_eval.png')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'output_dir')
