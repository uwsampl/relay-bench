import argparse
import os
from collections import OrderedDict

from common import (write_status, prepare_out_file, time_difference,
                    sort_data, render_exception)
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType

OUR_NAME = 'InterNeuron'

def generate_nlp_comparisons(raw_data, output_dir):
    filename = 'nlp-comp-cpu.png'

    # empty data: nothing to do
    if not raw_data.items():
        return

    data = {
        'raw': raw_data,
        'meta': ['Framework', 'Network', f'Mean Inference Time Slowdown\nRelative to {OUR_NAME}']
    }

    builder = PlotBuilder()\
            .set_title('NLP') \
            .set_y_label(data['meta'][2]) \
            .set_y_scale(PlotScale.LINEAR) \
            .set_aspect_ratio(1.2) \
            .set_figure_height(4) \
            .set_sig_figs(2) \
            .set_bar_colors(['C2', 'C1']) \
            .set_unit_type(UnitType.COMPARATIVE) \
            .make(PlotType.MULTI_BAR, data)

    containers = builder.ax().containers
    texts = builder.ax().texts

    mxnet_container = containers[0]
    mxnet_texts = texts[0:3] + [None, None]
    for bar, text in zip(mxnet_container.get_children(), mxnet_texts):
        offs = bar.get_width() / 2
        bar.set_x(bar.get_x() + offs)
        if text is not None:
            text.set_x(text.get_position()[0] + offs)

    pytorch_container = containers[1]
    pytorch_texts = [None, None, None] + texts[3:]
    for bar, text in zip(pytorch_container.get_children(), pytorch_texts):
        offs = bar.get_width() / 2
        bar.set_x(bar.get_x() - offs)
        if text is not None:
            text.set_x(text.get_position()[0] - offs)

    builder.save(output_dir, filename)


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
            'CharRNN': 0.0,
            'TreeLSTM': 0.0,
         }),
        ('PyTorch', {
            'Gluon RNN': 0.0,
            'GRU': 0.0,
            'LSTM': 0.0,
            'CharRNN': raw_data['char_rnn']['cpu']['Pytorch'] / raw_data['char_rnn']['cpu']['Aot'],
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
