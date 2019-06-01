'''Simple script that reads the dashboard JSON data and posts it to a
Slack webhook that is passed in.'''
import argparse
import json
import os
import requests
import sys

def relay_opt_text_summary(data):
    ret = ''
    for (network, opt_times) in data.items():
        ret += '{}: '.format(network)
        ret += ', '.join(['{:.3f} ({})'.format(time*1e3, opt)
                          for (opt, time) in opt_times.items()])
        ret += '\n'
    return ret


def cnn_text_summary(data):
    ret = ''
    for (framework, net_times) in data.items():
        ret += '{}: '.format(framework)
        ret += ', '.join(['{:.3f} ({})'.format(time*1e3, net)
                          for (net, time) in net_times.items()])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='')
    parser.add_argument('--post-webhook', type=str, required=True)
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, 'data.json')) as json_file:
        data = json.load(json_file)
        message = {}
        message['attachments'] = [{
            'fallback': 'Dashboard data after run on {}'.format(data['timestamp']),
            'color': '#000000',
            'pretext': 'Dashboard data after run on {}. Times are in ms.'.format(data['timestamp']),
            'fields': [
                {
                    'title': 'Relay Optimization Level Comparisons (CPU)',
                    'value': relay_opt_text_summary(data['opt_cpu']),
                    'short': False
                },
                {
                    'title': 'Relay Optimization Level Comparisons (GPU)',
                    'value': relay_opt_text_summary(data['opt_gpu']),
                    'short': False
                },
                {
                    'title': 'CNN Comparisons (CPU)',
                    'value': cnn_text_summary(data['cnn_cpu']),
                    'short': False
                },
                {
                    'title': 'CNN Comparisons (GPU)',
                    'value': cnn_text_summary(data['cnn_gpu']),
                    'short': False
                },
                {
                    'title': 'Char RNN Comparison (CPU)',
                    'value': char_rnn_text_summary(data['char-rnn']),
                    'short': False
                },
                {
                    'title': 'TreeLSTM Comparison (CPU)',
                    'value': tree_lstm_text_summary(data['treelstm']),
                    'short': False
                }
            ]
        }]
        r = requests.post(args.post_webhook, json=message)
