import argparse
import itertools
import os
from collections import OrderedDict

import pandas as pd

from plot_util import PlotBuilder, PlotScale, PlotType, UnitType, to_dataframe
from common import traverse_fields

# Raspberry Pi 3
PI_DF_DATA = pd.DataFrame({
    'Network': (['ResNet-18'] * 3) + (['MobileNet'] * 3) + (['Inception'] * 3),
    'Quantization Scheme': ['float32', 'int8/int32', 'int8/int16'] * 3,
    'Time': [
        323, 288, 123,
        122, 136, 113,
        3205, 1195, 928,
    ]
})
PI_DICT_DATA = {
    'raw': OrderedDict([
        ('float32', OrderedDict([
            ('ResNet-18', 323),
            ('MobileNet', 122),
            ('Inception', 3205),
            ])),
        ('int8/int32', OrderedDict([
            ('ResNet-18', 288),
            ('MobileNet', 136),
            ('Inception', 1195),
            ])),
        ('int8/int16', OrderedDict([
            ('ResNet-18', 123),
            ('MobileNet', 113),
            ('Inception', 928),
            ]))
        ]),
    'meta': ['Quantization Scheme', 'Network', 'Time']
}

# RK3399
RK_DF_DATA = pd.DataFrame({
    'Network': (['ResNet-18'] * 3) + (['MobileNet'] * 3) + (['Inception'] * 3),
    'Quantization Scheme': ['float32', 'int8/int32', 'int8/int16'] * 3,
    'Time': [
        161, 184, 85,
        67, 78, 63,
        1286, 1071, 648
    ]
})
RK_DICT_DATA = {
    'raw': OrderedDict([
        ('float32', OrderedDict([
            ('ResNet-18', 161),
            ('MobileNet', 67),
            ('Inception', 1286),
            ])),
        ('int8/int32', OrderedDict([
            ('ResNet-18', 184),
            ('MobileNet', 78),
            ('Inception', 1071),
            ])),
        ('int8/int16', OrderedDict([
            ('ResNet-18', 85),
            ('MobileNet', 63),
            ('Inception', 648),
            ]))
        ]),
    'meta': ['Quantization Scheme', 'Network', 'Mean Inference Time (ms)']
}

def make_plot(title, data, output_dir, filename):
    PlotBuilder().set_title(title) \
                 .set_y_label(data['meta'][2]) \
                 .set_y_scale(PlotScale.LINEAR) \
                 .set_aspect_ratio(1.4) \
                 .set_figure_height(3) \
                 .set_font_scale(0.7) \
                 .set_unit_type(UnitType.MILLISECONDS) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)


def main(data_dir, output_dir):
    make_plot('Raspberry Pi 3', PI_DICT_DATA, output_dir, 'raspberry_pi_quantization.png')
    make_plot('RK3399', RK_DICT_DATA, output_dir, 'rk3399_quantization.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
