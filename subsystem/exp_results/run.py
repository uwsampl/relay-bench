'''Reads experiment summaries and posts them to Slack.'''
import argparse
import json
import os
import sys
import textwrap

from common import (check_file_exists, read_config,
                    read_json, write_status)
from slack_util import (generate_ping_list,
                        build_field, build_attachment, build_message,
                        post_message)

def failed_experiment_field(exp, failure_stage, status, notify):
    message = 'Failed at stage {}:\n{}'.format(
        failure_stage,
        textwrap.shorten(status['message'], width=280))

    if notify:
        message += '\nATTN: {}'.format(generate_ping_list(notify))

    return build_field(title=exp, value=message)


def main(config_dir, home_dir, output_dir):
    config = read_config(config_dir)
    if 'webhook_url' not in config:
        write_status(output_dir, False, 'No webhook URL given')
        sys.exit(1)

    webhook = config['webhook_url']
    description = ''
    if 'description' in config:
        description = config['description']

    # dashboard should have already run so all experiment
    # directories should exist and be in order
    exp_config_dir = os.path.join(home_dir, 'config', 'experiments')
    exp_status_dir = os.path.join(home_dir, 'results', 'experiments', 'status')
    exp_summary_dir = os.path.join(home_dir, 'results', 'experiments', 'summary')

    inactive_experiments = []
    # failed experiments: list of slack fields
    failed_experiments = []
    # successful experiments: list of slack fields
    successful_experiments = []
    # experiments on which visualization failed (just names)
    failed_graphs = []

    for subdir, _, _ in os.walk(exp_status_dir):
        if subdir == exp_status_dir:
            continue
        exp_name = os.path.basename(subdir)
        exp_conf = read_config(os.path.join(exp_config_dir, exp_name))

        precheck_status = read_json(subdir, 'precheck.json')
        if not precheck_status['success']:
            failed_experiments.append(
                failed_experiment_field(exp_name, 'precheck',
                                        precheck_status, []))
            continue

        exp_title = exp_name if 'title' not in exp_conf else exp_conf['title']
        notify = exp_conf['notify']
        if not exp_conf['active']:
            inactive_experiments.append(exp_title)
            continue

        failure = False
        if check_file_exists(subdir, 'setup.json'):
            setup_status = read_json(subdir, 'setup.json')
            if not setup_status['success']:
                failed_experiments.append(
                    failed_experiment_field(exp_name, 'setup',
                                            setup_status, notify))
                failure = True

        if failure:
            continue

        for stage in ['run', 'analysis', 'summary']:
            stage_status = read_json(subdir, stage + '.json')
            if not stage_status['success']:
                failed_experiments.append(
                    failed_experiment_field(exp_name, stage,
                                            stage_status, notify))
                failure = True
                break

        if failure:
            continue

        # failure to visualize is not as big a deal as failing to
        # run or analyze the experiment, so we only report it but
        # don't fail to report the summary
        visualization_status = read_json(subdir, 'visualization.json')
        if not visualization_status['success']:
            failed_graphs.append(exp_title)

        summary = read_json(os.path.join(exp_summary_dir, exp_name),
                            'summary.json')
        successful_experiments.append(
            build_field(summary['title'], summary['value']))

    # produce messages
    attachments = []
    if successful_experiments:
        attachments.append(
            build_attachment(
                title='Successful benchmarks',
                pretext=description,
                fields=successful_experiments))
    if failed_experiments:
        attachments.append(
            build_attachment(
                color='#fa0000',
                title='Failed benchmarks',
                fields=failed_experiments))
    if inactive_experiments:
        attachments.append(
            build_attachment(
                color='#616161',
                title='Inactive benchmarks',
                text=', '.join(inactive_experiments)))
    if failed_graphs:
        attachments.append(
            build_attachment(
                color='#fa0000',
                title='Failed to Visualize',
                text=', '.join(failed_graphs)))

    success, report = post_message(
        webhook,
        build_message(
            text='Dashboard Results',
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
