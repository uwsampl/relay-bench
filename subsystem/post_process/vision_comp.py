import os
from collections import OrderedDict

from common import (write_status, prepare_out_file, time_difference,
                    invoke_main, read_config, sort_data, render_exception)
from dashboard_info import DashboardInfo
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType
from check_prerequisites import check_prerequisites

def generate_vision_comparisons(our_name, raw_data, output_dir):
    filename = 'cnn-comp-gpu.png'

    # empty data: nothing to do
    if not raw_data.items():
        return

    data = {
        'raw': raw_data,
        'meta': ['Framework', 'Network', f'Mean Inference Time Speedup\nof {our_name}']
    }

    PlotBuilder().set_y_label(data['meta'][2]) \
                 .set_title('Vision') \
                 .set_aspect_ratio(1.55) \
                 .set_figure_height(4) \
                 .set_y_scale(PlotScale.LINEAR) \
                 .set_sig_figs(2) \
                 .set_unit_type(UnitType.COMPARATIVE) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)


def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    conf = read_config(config_dir)
    our_name = 'Relay'
    if 'our_name' in conf:
        our_name = conf['our_name']

    conf_fws = ['relay', 'pt', 'tf', 'mxnet', 'nnvm']
    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    prereqs, msg = check_prerequisites(info, {
        'cnn_comp': {
            'devices': ['gpu'],
            'use_xla': True,
            'networks': networks,
            'frameworks': conf_fws
        }
    })
    if not prereqs:
        write_status(output_dir, False, msg)
        return 1

    all_data = sort_data(info.exp_data_dir('cnn_comp'))
    raw_data = all_data[-1]['gpu']

    our_fw = 'Relay'
    other_fws = ['TensorFlow', 'Pytorch', 'MxNet',
                 'NNVM', 'TF XLA']
    fw_name_map = {fw: fw for fw in other_fws}
    fw_name_map['Pytorch'] = 'PyTorch'

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    network_name_map = {
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet V2',
        'nature-dqn': 'DQN',
        'vgg-16': 'VGG-16'
    }

    plot_data = OrderedDict([
        (fw_name_map[fw], {
            network_name_map[network]: raw_data[fw][network] / raw_data[our_fw][network]
            for network in networks})
        for fw in other_fws
    ])

    try:
        generate_vision_comparisons(our_name, plot_data, output_dir)
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        return 1

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
