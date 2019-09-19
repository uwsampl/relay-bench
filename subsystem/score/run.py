import os
from collections import OrderedDict

from common import (write_status, write_json, prepare_out_file,
                    get_timestamp, idemp_mkdir,
                    time_difference, invoke_main, read_config,
                    sort_data, render_exception)
from dashboard_info import DashboardInfo
from check_prerequisites import check_prerequisites

from score_metrics import ScoreMetric, NNVMScore, RNNScore

SCORE_METRICS = {
    'nnvm_score': NNVMScore,
    'rnn_score': RNNScore
}

def format_scores(scores):
    return '\n'.join([msg for msg in scores.values()])


def process_score(info, score_metric, data_dir, graph_dir, timestamp):
    data = score_metric.compute_score(info)
    data['timestamp'] = timestamp
    write_json(data_dir, 'data_{}.json'.format(timestamp), data)

    # graphs failing is not a fatal error, just an inconvenience
    try:
        score_metric.score_graph(data, graph_dir)
        all_data = sort_data(data_dir)
        score_metric.longitudinal_graphs(all_data, graph_dir)
    except Exception as e:
        print(render_exception(e))
    finally:
        return score_metric.score_text(data)


def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    conf = read_config(config_dir)

    data_dir = os.path.join(output_dir, 'data')
    graph_dir = os.path.join(output_dir, 'graphs')
    idemp_mkdir(data_dir)
    idemp_mkdir(graph_dir)

    timestamp = get_timestamp()

    score_confs = conf['score_confs']
    metrics = set(score_confs.keys())
    metrics = metrics.intersection(set(SCORE_METRICS.keys()))

    if not metrics:
        write_status(output_dir, True, 'No scores to report')
        return 0

    score_data = {}
    score_reports = {}
    for metric in metrics:
        score_metric = SCORE_METRICS[metric](score_confs[metric])
        valid, msg = check_prerequisites(info, score_metric.prereq())
        if not valid:
            write_status(output_dir, False, msg)
            return 1

        score_data_dir = os.path.join(data_dir, metric)
        score_graph_dir = os.path.join(graph_dir, metric)
        idemp_mkdir(score_data_dir)
        idemp_mkdir(score_graph_dir)

        try:
            report = process_score(info, score_metric,
                                   score_data_dir, score_graph_dir,
                                   timestamp)
            score_reports[metric] = report
        except Exception as e:
            write_status(output_dir, False,
                         'Encountered exception while scoring {}:\n{}'.format(metric,
                                                                              render_exception(e)))
            return 1

    report = {
        'title': 'Metric Scores',
        'value': format_scores(score_reports)
    }
    write_json(output_dir, 'report.json', report)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
