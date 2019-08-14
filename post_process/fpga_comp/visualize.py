import argparse
import itertools
import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from plot_util import PlotBuilder, PlotScale, PlotType, UnitType, to_dataframe
from common import traverse_fields

sns.set(style='darkgrid')
sns.set_context('paper')

#DF_DATA = pd.DataFrame({
#    'Network': (['ResNet-18'] * 3) + (['MobileNet'] * 3) + (['Inception'] * 3),
#    'Quantization Scheme': ['float32', 'int8/int32', 'int8/int16'] * 3,
#    'Time': [
#        323, 288, 123,
#        122, 136, 113,
#        3205, 1195, 928,
#    ]
#})
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
