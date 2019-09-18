"""
Data structures handling logic for computing and rendering
dashboard score metrics.
"""
import os

from common import sort_data, time_difference
from plot_util import PlotBuilder, PlotScale, PlotType, generate_longitudinal_comparisons
from dashboard_info import DashboardInfo

def latest_data(info, exp, dev):
    return sort_data(info.exp_data_dir(exp))[-1][dev]


# plotbuilder dislikes ints
def make_ratio_score(score, include_timestamp=False):
    ret = {'score': float(score['Score'] / score['Total'])}
    if include_timestamp:
        ret['timestamp'] = score['timestamp']
    return ret


class ScoreMetric:
    def __init__(self, config):
        self._prereq = {}


    def prereq(self):
        """
        Return a set of experiment config prerequisites
        {exp_name : { config_field : expected_value  }}
        """
        return self._prereq


    def compute_score(self, info):
        """
        Given a DashboardInfo structure, compute a score.
        """
        return 0


    def score_text(self, score):
        """
        Return a text message describing the score
        """
        return str(score)


    def score_graph(self, score, graph_dir):
        """
        Save a graph visualizing the score in the given directory.
        """
        return


    def longitudinal_graphs(self, scores, graph_dir):
        """
        Given a sorted list of all scores, save longitudinal
        graphs in the given directory.
        """
        most_recent = scores[-1]
        last_two_weeks = [entry for entry in scores
                          if time_difference(most_recent, entry).days < 14]

        generate_longitudinal_comparisons(scores, graph_dir,
                                          'all_time', stat_name='Score')
        generate_longitudinal_comparisons(last_two_weeks, graph_dir,
                                          'two_weeks', stat_name='Score')


class NNVMScore(ScoreMetric):
    def __init__(self, config):
        self.margin = 0
        if 'margin' in config:
            self.margin = config['margin']
        self._prereq = {
            'cnn_comp': {
                'devices': ['gpu'],
                'frameworks': ['relay', 'nnvm']
            }
        }


    def compute_score(self, info):
        eps = 1e-6
        margin = self.margin + eps
        raw_data = latest_data(info, 'cnn_comp', 'gpu')

        conf = info.read_exp_config('cnn_comp')
        total = len(conf['networks'])
        score = len([
            network for network in conf['networks']
            if raw_data['Relay'][network] < raw_data['NNVM'][network] - margin
        ])
        return {
            'Score': score,
            'Total': total
        }


    def score_text(self, score):
        return 'Beat NNVM: {} of {} networks'.format(score['Score'], score['Total'])


    def score_graph(self, score, graph_dir):

        data = {
            'raw': make_ratio_score(score),
            'meta': ['Score', 'Ratio']
        }

        dest_dir = os.path.join(graph_dir, 'comparison')
        PlotBuilder().set_title('NNVM Score') \
                     .set_y_label('Number of Benchmarks') \
                     .make(PlotType.BAR, data) \
                     .save(dest_dir, 'nnvm_score.png')


    def longitudinal_graphs(self, scores, graph_dir):
        map_scores = [
            make_ratio_score(score, include_timestamp=True)
            for score in scores
        ]
        super().longitudinal_graphs(map_scores, graph_dir)


class RNNScore(ScoreMetric):
    def __init__(self, config):
        self.speedup = 2
        if 'speedup' in config:
            self.speedup = config['speedup']

        self._prereq = {
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


    def compute_score(self, info):
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

        score = len([ratio for ratio in ratios if ratio >= self.speedup])
        total = len(ratios)
        return {
            'Score': score,
            'Total': total
        }


    def score_text(self, score):
        return '{}x speedup on NLP: {} of {} RNNs'.format(
            self.speedup, score['Score'], score['Total'])


    def score_graph(self, score, graph_dir):
        data = {
            'raw': make_ratio_score(score),
            'meta': ['Score', 'Ratio']
        }

        dest_dir = os.path.join(graph_dir, 'comparison')
        PlotBuilder().set_title('RNN Speedup') \
                     .set_y_label('Number of Benchmarks') \
                     .make(PlotType.BAR, data) \
                     .save(dest_dir, 'rnn_score.png')


    def longitudinal_graphs(self, scores, graph_dir):
        map_scores = [
            make_ratio_score(score, include_timestamp=True)
            for score in scores
        ]
        super().longitudinal_graphs(map_scores, graph_dir)
