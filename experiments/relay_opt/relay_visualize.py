import os

import numpy as np

from validate_config import validate
from common import (invoke_main, write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import PlotBuilder, PlotScale, PlotType, generate_longitudinal_comparisons

MODEL_TO_TEXT = {
    'nature-dqn': 'Nature DQN',
    'vgg-16': 'VGG-16',
    'resnet-18': 'ResNet-18',
    'mobilenet': 'MobileNet'
}

def generate_relay_opt_comparisons(title, filename, raw_data, networks, output_prefix=''):
    comparison_dir = os.path.join(output_prefix, 'comparison')

    # empty data: nothing to do
    if not raw_data.items():
        return

    # make model names presentable
    for (opt_level, models) in raw_data.items():
        # NOTE: need to convert the keys to a list, since we're mutating them
        # during traversal.
        for model in list(models.keys()):
            val = models[model]
            del models[model]
            models[MODEL_TO_TEXT[model]] = val

    data = {
        'raw': raw_data,
        'meta': ['Opt Level', 'Network', 'Mean Inference Time (ms)'],
    }

    PlotBuilder().set_title(title) \
                 .set_y_label('Time (ms)') \
                 .set_y_scale(PlotScale.LOG) \
                 .set_figure_height(3) \
                 .set_aspect_ratio(3.3) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(comparison_dir, filename)


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    devs = config['devices']
    networks = config['networks']

    # read in data, output graphs of most recent data, and output longitudinal graphs
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    last_two_weeks = [entry for entry in all_data
                      if time_difference(most_recent, entry).days < 14]

    try:
        generate_longitudinal_comparisons(all_data, output_dir, 'all_time')
        generate_longitudinal_comparisons(last_two_weeks, output_dir, 'two_weeks')
        for dev in devs:
            generate_relay_opt_comparisons('Relay CNN Opt Level on {}'.format(dev.upper()),
                                           'relay-cnn-{}.png'.format(dev), most_recent[dev],
                                           networks, output_dir)

    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        return 1

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
