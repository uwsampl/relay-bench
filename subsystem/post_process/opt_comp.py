import os
from collections import OrderedDict

from common import (write_status, prepare_out_file, time_difference,
                    invoke_main, sort_data, render_exception)
from dashboard_info import DashboardInfo
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType
from check_prerequisites import check_prerequisites

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
                 .set_aspect_ratio(1.55) \
                 .set_figure_height(4) \
                 .set_sig_figs(3) \
                 .set_unit_type(UnitType.COMPARATIVE) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)


def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    prereqs, msg = check_prerequisites(info, {
        'relay_opt': {
            'devices': ['gpu'],
            'opt_levels': [0,1,2,3,4],
            'networks': networks
        }
    })
    if not prereqs:
        write_status(output_dir, False, msg)
        return 1

    all_data = sort_data(info.exp_data_dir('relay_opt'))
    raw_data = all_data[-1]['gpu']

    baseline = 'O0'
    opts = ['O1', 'O2', 'O3', 'O4']

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
        return 1

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
