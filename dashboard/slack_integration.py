'''Simple script that reads the dashboard JSON data and posts it to a
Slack webhook that is passed in.'''
import argparse
import json
import math
import os
import requests
import sys

def relay_opt_text_summary(data):
    if len(data.items()) == 0:
        return ''
    opts = [opt for (opt, _) in data[list(data.keys())[0]].items()]
    ret = 'Format: ({})\n'.format(', '.join(opts))
    for (network, opt_times) in data.items():
        ret += '{}: '.format(network)
        ret += ', '.join(['{:.3f}'.format(time*1e3)
                          for (_, time) in opt_times.items()])
        ret += '\n'
    return ret


def cnn_text_summary(data):
    if len(data.items()) == 0:
        return ''
    nets = [net for (net, _) in data[list(data.keys())[0]].items()]
    ret = 'Format: ({})\n'.format(', '.join(nets))
    for (framework, net_times) in data.items():
        ret += '{}: '.format(framework)
        ret += ', '.join(['{:.3f}'.format(time*1e3)
                          for (_, time) in net_times.items()])
        ret += '\n'
    return ret


def char_rnn_text_summary(data):
    ret = ''
    for (framework, time) in data.items():
        ret += '{}: {:.3f}\n'.format(framework, time*1e3)
    return ret


def tree_lstm_text_summary(data):
    ret = ''
    for (framework, time) in data.items():
        ret += '{}: {:.3f}\n'.format(framework, time*1e3)
    return ret


def nans_present(data, fields):
    for field in fields:
        for (_, val) in data[field].items():
            if not isinstance(val, dict):
                if math.isnan(val):
                    return True
                continue
            for (_, time) in val.items():
                if math.isnan(time):
                    return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--ping-users', type=str, default='',
                        help='Comma-separated list of user IDs to ping if there is a problem.')
    parser.add_argument('--post-webhook', type=str, required=True)
    args = parser.parse_args()

    benchmarks = {
        'opt_cpu': ('Relay Optimization Level Comparisons (CPU)', relay_opt_text_summary),
        'opt_gpu': ('Relay Optimization Level Comparisons (GPU)', relay_opt_text_summary),
        'cnn_cpu': ('CNN Comparisons (CPU)', cnn_text_summary),
        'cnn_gpu': ('CNN Comparisons (GPU)', cnn_text_summary),
        'char-rnn': ('Char RNN Comparison (CPU)', char_rnn_text_summary),
        'treelstm': ('TreeLSTM Comparison (CPU)', tree_lstm_text_summary)
    }

    with open(os.path.join(args.data_dir, 'data.json')) as json_file:
        data = json.load(json_file)
        message = {}
        message['attachments'] = [{
            'fallback': 'Dashboard data after run on {}'.format(data['timestamp']),
            'color': '#000000',
            'pretext': 'Dashboard data after run on {}. Times are in ms.'.format(data['timestamp']),
            'fields': [
                {
                    'title': title,
                    'value': summary(data[field]),
                    'short': False
                }
                for (field, (title, summary)) in benchmarks.items()
            ]
        }]
        r = requests.post(args.post_webhook, json=message)

        # ping if there's a NaN in the data
        if nans_present(data, benchmarks.keys()):
            users = args.ping_users.split(',')
            if not users:
                # no point in writing a warning if no one will be pinged
                sys.exit()
            pings = ', '.join(['<@{}>'.format(user) for user in users])
            message = {
                'text': 'Attention {}: something is broken!'.format(pings),
                'attachments': [{
                    'color': '#fa0000',
                    'title': 'Failing benchmarks',
                    'text': ', '.join([title for (field, (title, _)) in benchmarks.items() if nans_present(data, [field])]),
                    'fields': []
                }]
            }
            r = requests.post(args.post_webhook, json=message)
