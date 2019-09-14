import os
import sys
from collections import OrderedDict

from common import (write_status, write_json, prepare_out_file, time_difference,
                    invoke_main, read_config, sort_data, render_exception)
from dashboard_info import DashboardInfo
from check_prerequisites import check_prerequisites

def latest_data(info, exp, dev):
    return sort_data(info.exp_data_dir(exp))[-1][dev]


def nnvm_score():
    prereq = {
        'cnn_comp': {
            'devices': ['gpu'],
            'frameworks': ['relay', 'nnvm']
        }
    }

    def score_func(info):
        eps = 1e-6
        raw_data = latest_data(info, 'cnn_comp', 'gpu')

        conf = info.read_exp_config('cnn_comp')
        total = len(conf['networks'])
        score = len([
            network for network in conf['networks']
            if raw_data['Relay'][network] < raw_data['NNVM'][network] - eps
        ])
        return (score, total)

    def render_func(pair):
        return 'Beat NNVM: {} of {} networks'.format(pair[0], pair[1])

    return prereq, score_func, render_func


def rnn_score():
    prereq = {
        'gluon_rnns': {
            'frameworks': ['relay', 'mxnet'],
            'relay_methods': ['aot']
        },
        'char_rnn': {
            'frameworks': ['relay', 'pt'],
            'relay_methods': ['aot'],
            'relay_configs': ['loop']
        },
        'treelstm': {
            'frameworks': ['relay', 'pt'],
            'relay_methods': ['aot']
        }
    }

    def score_func(info):
        raw_gluon_data = latest_data(info, 'gluon_rnns', 'cpu')
        raw_char_data = latest_data(info, 'char_rnn', 'cpu')
        raw_tlstm_data = latest_data(info, 'treelstm', 'cpu')
        gluon_conf = info.read_exp_config('gluon_rnns')

        ratios = [
            *[raw_gluon_data['MxNet'][network] / raw_gluon_data['Aot'][network]
              for network in gluon_conf['networks']],
            raw_char_data['Pytorch'] / raw_char_data['Aot'],
            raw_tlstm_data['Pytorch'] / raw_tlstm_data['Aot']
        ]

        score = len([ratio for ratio in ratios if ratio >= 2])
        total = len(ratios)
        return (score, total)


    def render_func(pair):
        return '2x speedup on NLP: {} of {} RNNs'.format(pair[0], pair[1])

    return prereq, score_func, render_func


def format_scores(scores):
    return '\n'.join([msg for _, msg in scores.values()])


def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    conf = read_config(config_dir)

    score_metrics = set(conf['metrics'])
    # metric -> (prereq dict, score func, render func)
    metric_info = {'beat_nnvm': nnvm_score(), 'rnn_2x': rnn_score()}

    score_metrics = score_metrics.intersection(set(metric_info.keys()))

    if not score_metrics:
        write_status(output_dir, True, 'No scores to report')
        sys.exit(0)

    scores = {}
    for metric in score_metrics:
        prereq, score_func, render_func = metric_info[metric]
        valid, msg = check_prerequisites(info, prereq)
        if not valid:
            write_status(output_dir, False, msg)
            sys.exit(1)

        try:
            score = score_func(info)
            scores[metric] = (score, render_func(score))
        except Exception as e:
            write_status(output_dir, False,
                         'Encountered excpetion while scoring {}:\n{}'.format(metric,
                                                                              render_exception(e)))
            sys.exit(1)

    report = {
        'title': 'Metric Scores',
        'value': format_scores(scores)
    }
    write_json(output_dir, 'report.json', report)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
