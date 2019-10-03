"""
Checks subsystem output subdirectories for a report.json
and, if present, posts to slack
"""
import argparse
import json
import os
import textwrap

from common import (invoke_main, read_config, write_status,
                    check_file_exists, read_json)
from dashboard_info import DashboardInfo
from slack_util import (build_field, build_attachment, build_message,
                        post_message)

def failed_subsys_field(subsys, status):
    message = 'Failed to run:\n{}'.format(
        textwrap.shorten(status['message'], width=280))
    return build_field(title=subsys, value=message)


def main(config_dir, home_dir, output_dir):
    config = read_config(config_dir)
    if 'webhook_url' not in config:
        write_status(output_dir, False, 'No webhook URL given')
        return 1

    webhook = config['webhook_url']

    info = DashboardInfo(home_dir)

    failed_subsys = []
    reports = []
    failed_reports = []

    for subsys in info.all_present_subsystems():
        # ignore self
        if subsys == 'subsys_reporter':
            continue

        if not info.subsys_active(subsys):
            continue

        status = info.subsys_stage_status(subsys, 'run')
        if not status['success']:
            failed_subsys.append(failed_subsys_field(subsys, status))
            continue

        report_present = check_file_exists(info.subsys_output_dir(subsys), 'report.json')
        if not report_present:
            continue

        try:
            report = read_json(info.subsys_output_dir(subsys), 'report.json')
            reports.append(build_field(
                title=report['title'],
                value=report['value']))
        except Exception:
            failed_reports.append(subsys_name)

    attachments = []
    if reports:
        attachments.append(build_attachment(
            title='Reports',
            fields=reports))
    if failed_reports or failed_subsys:
        failure_text = ''
        if failed_reports:
            failure_text = 'Failed to parse reports: {}'.format(', '.join(failed_reports))
        attachments.append(build_attachment(
            title='Errors',
            text=failure_text,
            color='#fa0000',
            fields=failed_subsys))

    if not attachments:
        write_status(output_dir, True, 'Nothing to report')
        return 0

    success, msg = post_message(
        webhook,
        build_message(
            text='Subsystem Results',
            attachments=attachments))
    write_status(output_dir, success, msg)


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
