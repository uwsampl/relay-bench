import os

from validate_config import validate
from common import (invoke_main, write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import PlotBuilder, PlotScale, PlotType, generate_longitudinal_comparisons

def generate_gluon_rnn_comparison(title, filename, raw_data, output_prefix=''):
    if len(raw_data) == 0:
        return

    data = {
        'raw': raw_data,
        'meta': ['Executor', 'Network', 'Mean Inference Time (ms)'],
    }

    comparison_dir = os.path.join(output_prefix, 'comparison')
    PlotBuilder().set_title(title) \
                 .set_y_label(data['meta'][2]) \
                 .set_y_scale(PlotScale.LOG) \
                 .make(PlotType.MULTI_BAR, data) \
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
            generate_gluon_rnn_comparison('Gluon RNN Comparison on {}'.format(dev.upper()),
                                         'gluon_rnns-{}.png'.format(dev),
                                         most_recent[dev], output_dir)

    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        return 1

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
