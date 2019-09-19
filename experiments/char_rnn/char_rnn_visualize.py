import os

from validate_config import validate
from common import (invoke_main, write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import PlotScale, PlotType, PlotBuilder, generate_longitudinal_comparisons

def generate_char_rnn_comparison(title, filename, raw_data, output_prefix=''):
    data = {
        'raw': raw_data,
        'meta': ['Executor', 'Mean Inference Time (ms)']
    }

    comparison_dir = os.path.join(output_prefix, 'comparison')
    PlotBuilder().set_title(title) \
                 .set_x_label(data['meta'][0]) \
                 .set_y_label(data['meta'][1]) \
                 .set_y_scale(PlotScale.LOG) \
                 .make(PlotType.BAR, data) \
                 .save(comparison_dir, filename)


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    devs = config['devices']

    # read in data, output graphs of most recent data, and output longitudinal graphs
    all_data = sort_data(data_dir)
    most_recent = all_data[-1]

    last_two_weeks = [entry for entry in all_data
                      if time_difference(most_recent, entry).days < 14]

    try:
        generate_longitudinal_comparisons(all_data, output_dir, 'all_time')
        generate_longitudinal_comparisons(last_two_weeks, output_dir, 'two_weeks')
        for dev in devs:
            generate_char_rnn_comparison('Char RNN Comparison on {}'.format(dev.upper()),
                                         'char_rnn-{}.png'.format(dev),
                                         most_recent[dev], output_dir)

    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        return 1

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
