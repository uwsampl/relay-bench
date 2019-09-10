import argparse
import os
from collections import OrderedDict

from common import (write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from dashboard_info import DashboardInfo
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType
from check_prerequisites import check_prerequisites

def generate_pass_comparisons(raw_data, output_dir, filename):
    # empty data: nothing to do
    if not raw_data.items():
        return

    data = {
        'raw': raw_data,
        'meta': ['Pass Combo', 'Network', 'Mean Inference Time Speedup\nRelative to Baseline']
    }

    PlotBuilder().set_y_label(data['meta'][2]) \
                 .set_y_scale(PlotScale.LINEAR) \
                 .set_aspect_ratio(3.3) \
                 .set_figure_height(3) \
                 .set_font_scale(0.7) \
                 .set_sig_figs(3) \
                 .set_unit_type(UnitType.COMPARATIVE) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)

def parse_pass_combo(combo):
    [level, passes] = combo.split(';')
    pass_list = passes.split('|')
    return (level, tuple(pass_list))


def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    pass_spec_name_map = {
        '3;FuseOps': 'Op Fusion',
        '3;FoldConstant|FuseOps': '... + Constant Folding',
        '3;EliminateCommonSubexpr|FoldConstant|FuseOps': '... + Common Subexpr Elim',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldConstant|FuseOps': '... + Parallel Conv Comb',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|FoldConstant|FuseOps': '... + Axis Scale Folding',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|FoldConstant|FuseOps': '... + Cast Canonicalization',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|FoldConstant|FuseOps': '... + Op Canonicalization',
        '3;EliminateCommonSubexpr|CombineParallelConv2D|FoldScaleAxis|CanonicalizeCast|CanonicalizeOps|AlterOpLayout|FoldConstant|FuseOps': '... + Op Layout Alteration'
    }

    prereqs, msg = check_prerequisites(info, {
        'pass_comparison': {
            'networks': networks,
            'passes': [
                parse_combo(combo) for combo in pass_spec_name_map.keys()
            ]
        }
    })

    all_data = sort_data(info.exp_data_dir('pass_comparison'))
    raw_data = all_data[-1]

    baseline = '0;'

    network_name_map = {
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet V2',
        'nature-dqn': 'DQN',
        'vgg-16': 'VGG-16'
    }

    del raw_data['timestamp']
    del raw_data['tvm_hash']

    try:
        for (dev, raw_dev_data) in raw_data.items():
            plot_data = OrderedDict([
                (pass_spec_name_map[pass_spec], {
                    network_name_map[network]:
                    raw_dev_data[baseline][network] / raw_dev_data[pass_spec][network]
                    for network in networks})
                for pass_spec in pass_spec_name_map.keys()
            ])
            generate_pass_comparisons(plot_data, output_dir, f'pass-comp-{dev}.png')
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        sys.exit(1)

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--home-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.home_dir, args.output_dir)
