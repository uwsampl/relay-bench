import argparse
import os
from collections import OrderedDict

from common import (write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType

OUR_NAME = 'InterNeuron'

def generate_vision_comparisons(data, output_dir):
    filename = 'cnn-comp-gpu.png'

    # empty data: nothing to do
    if not data.items():
        return

    PlotBuilder().set_y_label(f'Mean Inference Time Slowdown Relative to {OUR_NAME}') \
                 .set_y_scale(PlotScale.LOG) \
                 .set_bar_width(0.15) \
                 .set_unit_type(UnitType.COMPARATIVE) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)


def main(data_dir, output_dir):
    cnn_comp_dir = os.path.join(data_dir, 'cnn_comp')
    all_data = sort_data(cnn_comp_dir)
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
        generate_vision_comparisons(plot_data, output_dir)
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
