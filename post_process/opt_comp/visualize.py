import argparse
import os
from collections import OrderedDict

from common import (write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType

def generate_opt_comparisons(raw_data, output_dir):
    filename = 'opt-comp-gpu.png'

    # empty data: nothing to do
    if not raw_data.items():
        return

    data = {
        'raw': raw_data,
        'meta': ['O Level', 'Network', 'Mean Inference Time Speedup Relative to\nNo Optimizations (O0)']
    }

    PlotBuilder().set_y_label(f'Mean Inference Time Speedup Relative to\nNo Optimizations (O0)') \
                 .set_y_scale(PlotScale.LINEAR) \
                 .set_bar_width(0.45) \
                 .set_sig_figs(3) \
                 .set_figsize((13, 6)) \
                 .set_unit_type(UnitType.COMPARATIVE) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)


def main(data_dir, output_dir):
    opt_comp_dir = os.path.join(data_dir, 'relay_opt')
    all_data = sort_data(opt_comp_dir)
    raw_data = all_data[-1]['gpu']

    baseline = 'O0'
    opts = ['O1', 'O2', 'O3']

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    network_name_map = {
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet V2',
        'nature-dqn': 'DQN',
        'vgg-16': 'VGG-16'
    }

    plot_data = OrderedDict([
        (opt, {
            network_name_map[network]: raw_data[baseline][network] / raw_data[opt][network]
            for network in networks})
        for opt in opts
    ])

    try:
        generate_opt_comparisons(plot_data, output_dir)
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        return

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
