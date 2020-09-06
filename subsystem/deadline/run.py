"""
Reports deadlines to Slack.
"""
import datetime
import json
import os

from common import invoke_main, read_config, write_status
from slack_util import (generate_ping_list,
                        build_field, build_attachment, build_message,
                        post_message, new_client)


def main(config_dir, home_dir, output_dir):
    config = read_config(config_dir)
    if 'channel_id' not in config:
        write_status(output_dir, False, 'No channel token given')
        return 1

    channel = config['channel_id']

    success, msg, client = new_client(config)

    if not success:
        write_status(output_dir, False, msg)
        return 1

    if 'deadlines' not in config:
        write_status(output_dir, True, 'No deadlines to report')
        return 1

    deadlines = config['deadlines']
    if not isinstance(deadlines, dict):
        write_status(output_dir, False, 'Invalid deadlines structure')
        return 0

    attachments = []
    present = datetime.datetime.now()
    for (name, info) in deadlines.items():
        if 'date' not in info:
            write_status(output_dir, False,
                         'Date missing in entry {} under {}'.format(info, name))
            return 1

        date = None
        try:
            date = datetime.datetime.strptime(info['date'], '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            write_status(output_dir, False,
                         'Could not parse date {}'.format(info['date']))
            return 1

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
        return 0

    success, _, report = post_message(
        client,
        channel,
        build_message(
            text='*Upcoming Deadlines*',
            attachments=attachments))
    write_status(output_dir, success, report)


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
