import argparse
import os
from collections import OrderedDict

from common import (write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType

def generate_pass_comparisons(data, output_dir):
    filename = 'pass-comp-gpu.png'

    # empty data: nothing to do
    if not data.items():
        return

    PlotBuilder().set_y_label(f'Mean Inference Time Speedup Relative to Baseline') \
                 .set_y_scale(PlotScale.LINEAR) \
                 .set_bar_width(0.05) \
                 .set_unit_type(UnitType.COMPARATIVE) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)


def main(data_dir, output_dir):
    pass_comp_dir = os.path.join(data_dir, 'pass_comparison')
    all_data = sort_data(pass_comp_dir)
    raw_data = all_data[-1]['gpu']

    our_fw = 'Relay'
    other_fws = ['TensorFlow', 'Pytorch', 'MxNet',
                 'NNVM', 'TF XLA']
    fw_name_map = {fw: fw for fw in other_fws}
    fw_name_map['Pytorch'] = 'PyTorch'

    baseline = 'Baseline'

    pass_lists = [
        'FuseOps',
        'FoldConstant|FuseOps',
        'EliminateCommonSubexpr|FoldConstant|FuseOps',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FuseOps',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FoldScaleAxis|FoldConstant|FuseOps',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FoldScaleAxis|CanonicalizeCast|FoldConstant|FuseOps',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|FoldConstant|FuseOps',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|AlterOpLayout|FoldConstant|FuseOps'
    ]

    pass_list_name_map = {
        'FuseOps': '+Fusion',
        'FoldConstant|FuseOps': '+FoldConstant',
        'EliminateCommonSubexpr|FoldConstant|FuseOps': '+EliminateCommonSubexpr',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FuseOps': '+CombineParallelConv2d',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FoldScaleAxis|FoldConstant|FuseOps': '+FoldScaleAxis',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FoldScaleAxis|CanonicalizeCast|FoldConstant|FuseOps': '+CanonicalizeCast',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|FoldConstant|FuseOps': '+CanonicalizeOps',
        'EliminateCommonSubexpr|CombineParallelConv2d|FoldConstant|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|AlterOpLayout|FoldConstant|FuseOps': '+AlterOpLayout'
    }

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    network_name_map = {
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet V2',
        'nature-dqn': 'DQN',
        'vgg-16': 'VGG-16'
    }

    plot_data = OrderedDict([
        (pass_list_name_map[pass_list], {
            network_name_map[network]: raw_data[baseline][network] / raw_data[pass_list][network]
            for network in networks})
        for pass_list in pass_lists
    ])

    try:
        generate_pass_comparisons(plot_data, output_dir)
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
