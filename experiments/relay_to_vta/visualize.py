import argparse
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

import numpy as np

from validate_config import validate
from common import (write_status, prepare_out_file, parse_timestamp,
                    sort_data, render_exception)
from plot_util import PlotBuilder, PlotScale, PlotType, generate_longitudinal_comparisons

SIM_TARGETS = {'sim', 'tsim'}
PHYS_TARGETS = {'pynq'}
MODEL_TO_TEXT = {
    'resnet18_v1': 'ResNet-18'
}
DEVICE_TO_TEXT = {
    'arm_cpu': 'Mobile CPU',
    'vta': 'Mobile CPU w/ FPGA'
}
METADATA_KEYS = {'timestamp', 'tvm_hash'}

def generate_arm_vta_comparisons(data, output_prefix):
    comparison_dir = os.path.join(output_prefix, 'comparison')

    plot_data = {}
    for (model, targets) in data.items():
        if model in METADATA_KEYS:
            continue
        model = MODEL_TO_TEXT[model]
        phys_targets = {target: v for (target, v) in targets.items() if target in PHYS_TARGETS}
        for (target, devices) in phys_targets.items():
            for (device, mean_time) in devices.items():
                # convert the device name to a readable description
                device = DEVICE_TO_TEXT[device]
                if target not in plot_data:
                    plot_data[target] = {}
                if device not in plot_data[target]:
                    plot_data[target][device] = {}
                if model not in plot_data[target][device]:
                    plot_data[target][device][model] = mean_time

    for (target, target_plot_data) in plot_data.items():
        filename = 'arm-vta-{}.png'.format(target)
        PlotBuilder().set_y_label('Inference Time (ms)') \
                     .set_y_scale(PlotScale.LOG) \
                     .make(PlotType.MULTI_BAR, target_plot_data) \
                     .save(comparison_dir, filename)


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    # read in data, output graphs of most recent data, and output longitudinal graphs
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    try:
        generate_longitudinal_comparisons(all_data, output_dir)
        generate_arm_vta_comparisons(most_recent, output_dir)
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        return

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.config_dir, args.output_dir)
