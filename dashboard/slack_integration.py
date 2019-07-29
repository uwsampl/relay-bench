'''Reads experiment summaries and posts them to Slack.'''
import argparse
import datetime
import json
import math
import os
import requests
import sys
import textwrap

from common import check_file_exists, read_json

def generate_ping_list(user_ids):
    return ', '.join(['<@{}>'.format(user_id) for user_id in user_ids])


def main(home_dir):
    if not check_file_exists(home_dir, 'config.json'):
        print('Dashboard config (config.json) is missing in {}'.format(home_dir))
        sys.exit(1)

    config = read_json(home_dir, 'config.json')
    if 'webhook_url' not in config:
        print('No Slack webhook given in dashboard config in {}'.format(home_dir))
        sys.exit(1)

    webhook = config['webhook_url']

    # in principle the dashboard should have already run
    # so status directories for the benchmarks should all exist
    config_dir = os.path.join(home_dir, 'config')
    status_dir = os.path.join(home_dir, 'status')
    summary_dir = os.path.join(home_dir, 'summary')

    inactive_experiments = []
    # failed experiments: list of (exp_name, failure stage, failure reason, [people to notify])
    failed_experiments = []
    # successful experiments: list of summary objects
    successful_experiments = []

    for subdir, _, _ in os.walk(status_dir):
        if subdir == status_dir:
            continue
        exp_name = os.path.basename(subdir)
        exp_conf = read_json(os.path.join(config_dir, exp_name),
                             'config.json')

        precheck_status = read_json(subdir, 'precheck.json')
        if not precheck_status['success']:
            failed_experiments.append((exp_name, 'precheck',
                                       textwrap.shorten(
                                           precheck_status['message'],
                                           width=50),
                                       []))
            continue

        exp_title = exp_name if 'title' not in exp_conf else exp_conf['title']
        notify = exp_conf['notify']
        if not exp_conf['active']:
            inactive_experiments.append(exp_title)
            continue

        failure = False
        for stage in ['run', 'analysis', 'summary']:
            stage_status = read_json(subdir, stage + '.json')
            if not stage_status['success']:
                failed_experiments.append((exp_title, stage,
                                           textwrap.shorten(
                                               stage_status['message'],
                                               width=50),
                                           notify))
                failure = True
                break
        if failure:
            continue

        summary = read_json(os.path.join(summary_dir, exp_name),
                            'summary.json')
        summary['short'] = False
        successful_experiments.append(summary)

    # produce messages
    attachments = []
    if successful_experiments:
        attachments.append({
            'color': '#000000',
            'pretext': config['slack_description'] if 'slack_description' in config else '',
            'title': 'Successful benchmarks',
            'fields': successful_experiments
        })
    if failed_experiments:
        attachments.append({
            'color': '#fa0000',
            'title': 'Failed benchmarks',
            'fields': [{
                'title': exp_title,
                'value': 'Failed at stage {}:\n{}'.format(stage, reason, pings)
                + ('' if not pings else '\nATTN: {}'.format(generate_ping_list(pings))),
                'short': False
            } for (exp_title, stage, reason, pings) in failed_experiments]
        })
    if inactive_experiments:
        attachments.append({
            'color': '#616161',
            'title': 'Inactive benchmarks',
            'text': ', '.join(inactive_experiments),
            'fields': []
        })
    message = {
        'text': 'Dashboard Results',
        'attachments': attachments
    }
    r = requests.post(webhook, json=message)

    # handle deadline subsystem
    if 'deadlines' in config:
        present = datetime.datetime.now()
        message = {'text': '*Upcoming deadlines*'}
        attachments = []
        for (name, info) in config['deadlines'].items():
            if 'date' not in info:
                continue
            date = datetime.datetime.strptime(info['date'], '%Y-%m-%d %H:%M:%S')
            diff = date - present
            days_left = diff.days
            if days_left < 0:
                # elapsed, so forget it
                continue
            alert = days_left <= 7

            pings = generate_ping_list(info['ping'])
            fields = [
                {'value': '{} days, {:.2f} hours left'.format(diff.days, diff.seconds/3600),
                 'short': False}
            ]
            if alert and 'ping' in info:
                fields.append({'value': 'Beware {}!'.format(pings), 'short': False})

            attachment = {
                'color': '#fa0000' if alert else '#0fbf24',
                'title': name,
                'text': 'Deadline: {}'.format(info['date']),
                'fields': fields
            }
            attachments.append(attachment)

        message['attachments'] = attachments
        r = requests.post(webhook, json=message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home-dir', type=str, required=True,
                        help='Dashboard home directory')
    args = parser.parse_args()
    main(args.home_dir)
