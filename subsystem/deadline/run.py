'''Reads experiment summaries and posts them to Slack.'''
import argparse
import datetime
import json
import os
import sys

from common import read_config, write_status
from slack_util import (generate_ping_list,
                        build_field, build_attachment, build_message,
                        post_message)


def main(config_dir, home_dir, output_dir):
    config = read_config(config_dir)
    if 'webhook_url' not in config:
        write_status(output_dir, False, 'No webhook URL given')
        sys.exit(1)

    webhook = config['webhook_url']

    if 'deadlines' not in config:
        write_status(output_dir, True, 'No deadlines to report')
        sys.exit(1)

    deadlines = config['deadlines']
    if not isinstance(deadlines, dict):
        write_status(output_dir, False, 'Invalid deadlines structure')
        sys.exit(0)

    attachments = []
    present = datetime.datetime.now()
    for (name, info) in deadlines.items():
        if 'date' not in info:
            write_status(output_dir, False,
                         'Date missing in entry {} under {}'.format(info, name))
            sys.exit(1)

        date = None
        try:
            date = datetime.datetime.strptime(info['date'], '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            write_status(output_dir, False,
                         'Could not parse date {}'.format(info['date']))
            sys.exit(1)

        diff = date - present
        days_left = diff.days
        if days_left < 0:
            # elapsed, so forget it
            continue
        alert = days_left <= 7

        time_left_msg = '{} days, {:.2f} hours left'.format(diff.days, diff.seconds/3600)
        fields = [build_field(value=time_left_msg)]
        if alert and 'ping' in info:
            pings = generate_ping_list(info['ping'])
            fields.append(build_field(value='Beware {}!'.format(pings)))
        attachments.append(build_attachment(
            title=name,
            text='Deadline: {}'.format(info['date']),
            fields=fields,
            color='#fa0000' if alert else '#0fbf24'))

    if not attachments:
        write_status(output_dir, True, 'All deadlines elapsed')
        sys.exit(0)

    success, report = post_message(
        webhook,
        build_message(
            text='*Upcoming Deadlines*',
            attachments=attachments))
    write_status(output_dir, success, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str, required=True,
                        help='Directory containing a config file')
    parser.add_argument('--home-dir', type=str, required=True,
                        help='Dashboard home directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory for any output')
    args = parser.parse_args()
    main(args.config_dir, args.home_dir, args.output_dir)
