"""
Subsystem for detecting whether a particular measurement in an
experiment is more than a standard deviation off from
its historic mean and producing a report.
"""
import itertools

import numpy as np

from common import (write_status, write_json,
                    get_timestamp, invoke_main, read_config,
                    sort_data, traverse_fields, gather_stats)
from slack_util import generate_ping_list
from dashboard_info import DashboardInfo

def format_report(info, exp_alert, pings):
    ret = ''
    if pings:
        ret += 'ATTN: {}\n'.format(generate_ping_list(pings))
    for exp, alert_list in exp_alert.items():
        if not alert_list:
            continue

        conf = info.read_exp_config(exp)
        exp_name = conf['title'] if 'title' in conf else exp

        ret += 'Alerts for {}:\n'.format(exp_name)
        for (fields, mean, sd, current) in alert_list:
            field_str = ', '.join([str(field) for field in fields])
            ret += '    ({}): {:.2e} (mean: {:.2e} +/- {:.2e})\n'.format(field_str, current, mean, sd)
    return ret


def main(config_dir, home_dir, output_dir):
    info = DashboardInfo(home_dir)
    conf = read_config(config_dir)

    pings = conf['notify'] if 'notify' in conf else []

    # map: exp -> [(fields w/ high SD, historic mean, SD, current)]
    exp_alerts = {}
    for exp in info.all_present_experiments():
        if not info.exp_active(exp):
            continue

        # not this subsystem's job to report on failures
        stage_statuses = info.exp_stage_statuses(exp)
        if 'run' not in stage_statuses or 'analysis' not in stage_statuses:
            continue
        if not stage_statuses['analysis']['success']:
            continue

        all_data = sort_data(info.exp_data_dir(exp))
        if len(all_data) <= 1:
            continue

        exp_alerts[exp] = []
        most_recent = all_data[-1]
        past_data = all_data[:-1]

        field_values = traverse_fields(most_recent)
        for fields in itertools.product(*field_values):
            current_stat, _ = gather_stats([most_recent], fields)
            current = current_stat[0]
            past_stats, _ = gather_stats(past_data, fields)

            past_sd = np.std(past_stats)
            past_mean = np.mean(past_stats)
            if abs(current - past_mean) > past_sd:
                exp_alerts[exp].append((fields, past_mean, past_sd, current))

        if not exp_alerts[exp]:
            del exp_alerts[exp]


    if exp_alerts:
        report = {
            'title': 'High SD Alerts',
            'value': format_report(info, exp_alerts, pings)
        }
        write_json(output_dir, 'report.json', report)

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
