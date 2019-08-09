import argparse
import os
from collections import OrderedDict

from common import (write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType

OUR_NAME = 'InterNeuron'

def generate_nlp_comparisons(data, output_dir):
    filename = 'nlp-comp-cpu.png'

    # empty data: nothing to do
    if not data.items():
        return

    PlotBuilder().set_y_label(f'Mean Inference Time Slowdown Relative to {OUR_NAME}') \
                 .set_y_scale(PlotScale.LINEAR) \
                 .set_bar_width(0.5) \
                 .set_unit_type(UnitType.COMPARATIVE) \
                 .make(PlotType.MULTI_BAR, data) \
                 .save(output_dir, filename)


def main(data_dir, output_dir):
    EXPERIMENT_PREREQS = {'treelstm', 'char_rnn', 'gluon_rnns'}

    raw_data = {}
    for exp in EXPERIMENT_PREREQS:
        exp_data_dir = os.path.join(data_dir, exp)
        all_data = sort_data(exp_data_dir)
        raw_data[exp] = all_data[-1]

    plot_data = OrderedDict([
        ('MxNet', {
            'Gluon RNN': raw_data['gluon_rnns']['cpu']['MxNet']['rnn'] / raw_data['gluon_rnns']['cpu']['Aot']['rnn'],
            'GRU': raw_data['gluon_rnns']['cpu']['MxNet']['gru'] / raw_data['gluon_rnns']['cpu']['Aot']['gru'],
            'LSTM': raw_data['gluon_rnns']['cpu']['MxNet']['lstm'] / raw_data['gluon_rnns']['cpu']['Aot']['lstm'],
            'Char-RNN': 0.0,
            'TreeLSTM': 0.0,
         }),
        ('PyTorch', {
            'Gluon RNN': 0.0,
            'GRU': 0.0,
            'LSTM': 0.0,
            'Char-RNN': raw_data['char_rnn']['cpu']['Pytorch'] / raw_data['char_rnn']['cpu']['Aot'],
            'TreeLSTM': raw_data['treelstm']['cpu']['Pytorch'] / raw_data['treelstm']['cpu']['Aot'],
         }),
    ])

    try:
        generate_nlp_comparisons(plot_data, output_dir)
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
