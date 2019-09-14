import os
import sys
from collections import OrderedDict

from common import (invoke_main, write_status, prepare_out_file, time_difference,
                    read_config, sort_data, render_exception)
from dashboard_info import DashboardInfo
from plot_util import PlotBuilder, PlotScale, PlotType, UnitType
from check_prerequisites import check_prerequisites

def generate_nlp_comparisons(our_name, raw_data, output_dir):
    filename = 'nlp-comp-cpu.png'

    # empty data: nothing to do
    if not raw_data.items():
        return

    data = {
        'raw': raw_data,
        'meta': ['Framework', 'Network', f'Mean Inference Time Speedup\nof {our_name}']
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


def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    conf = read_config(config_dir)
    our_name = 'Relay'
    if 'our_name' in conf:
        our_name = conf['our_name']

    prereqs, msg = check_prerequisites(info, {
        'treelstm': {
            'devices': ['cpu'],
            'frameworks': ['relay', 'pt'],
            'relay_methods': ['aot']
        },
        'char_rnn': {
            'devices': ['cpu'],
            'frameworks': ['relay', 'pt'],
            'relay_methods': ['aot'],
            'relay_configs': ['loop']
        },
        'gluon_rnns': {
            'devices': ['cpu'],
            'frameworks': ['relay', 'mxnet'],
            'networks': ['rnn', 'lstm', 'gru'],
            'relay_methods': ['aot']
        }
    })
    if not prereqs:
        write_status(output_dir, False, msg)
        sys.exit(1)

    raw_data = {}
    for exp in EXPERIMENT_PREREQS:
        all_data = sort_data(info.exp_data_dir(exp))
        raw_data[exp] = all_data[-1]

    plot_data = OrderedDict([
        ('MxNet', {
            'RNN': raw_data['gluon_rnns']['cpu']['MxNet']['rnn'] / raw_data['gluon_rnns']['cpu']['Aot']['rnn'],
            'GRU': raw_data['gluon_rnns']['cpu']['MxNet']['gru'] / raw_data['gluon_rnns']['cpu']['Aot']['gru'],
            'LSTM': raw_data['gluon_rnns']['cpu']['MxNet']['lstm'] / raw_data['gluon_rnns']['cpu']['Aot']['lstm'],
            'CharRNN': 0.0,
            'TreeLSTM': 0.0,
         }),
        ('PyTorch', {
            'RNN': 0.0,
            'GRU': 0.0,
            'LSTM': 0.0,
            'CharRNN': raw_data['char_rnn']['cpu']['Pytorch'] / raw_data['char_rnn']['cpu']['Aot'],
            'TreeLSTM': raw_data['treelstm']['cpu']['Pytorch'] / raw_data['treelstm']['cpu']['Aot'],
         }),
    ])

    try:
        generate_nlp_comparisons(our_name, plot_data, output_dir)
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered:\n' + render_exception(e))
        sys.exit(1)

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
