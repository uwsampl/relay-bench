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

    baseline = '0;'

    pass_specs = [
        '3;',
        '3;FoldConstant',
        '3;EliminateCommonSubexpr|FoldConstant',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldConstant',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|FoldConstant',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|FoldConstant',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|FoldConstant',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|AlterOpLayout|FoldConstant'
    ]

    pass_spec_name_map = {
        '3;': '+Fusion',
        '3;FoldConstant': '+FoldConstant',
        '3;EliminateCommonSubexpr|FoldConstant': '+EliminateCommonSubexpr',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldConstant': '+CombineParallelConv2d',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|FoldConstant': '+FoldScaleAxis',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|FoldConstant': '+CanonicalizeCast',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|FoldConstant': '+CanonicalizeOps',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|AlterOpLayout|FoldConstant': '+AlterOpLayout'
    }

    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    network_name_map = {
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet V2',
        'nature-dqn': 'DQN',
        'vgg-16': 'VGG-16'
    }

    plot_data = OrderedDict([
        (pass_spec_name_map[pass_spec], {
            network_name_map[network]:
            raw_data[baseline][network] / raw_data[pass_spec][network]
            for network in networks})
        for pass_spec in pass_specs
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
