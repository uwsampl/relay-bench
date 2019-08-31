"""
Checks whether the required experiments for a post-process
script exist and have run.
"""
import os

from common import check_file_exists, read_json, read_config

def match_values(expected, actual):
    if expected == actual:
        return True

    # Turn lists into sets to eliminate concerns about order.
    # Note that we only care that everything we expect is present;
    # it doesn't matter if the actual set contains more
    if isinstance(expected, list) and isinstance(actual, list):
        expected_set = set(expected)
        actual_set = None
        if len(actual) != 0 and isinstance(actual[0], list):
            actual_set = set([tuple(v) for v in actual])
        else:
            actual_set = set(actual)

        for item in expected_set:
            if item not in actual_set:
                return False
        return True

    return False


def check_prerequisites(home_dir, required_confs):
    config_dir = os.path.join(home_dir, 'config', 'experiments')
    status_dir = os.path.join(home_dir, 'results', 'experiments', 'status')

    for (exp_name, conf_entries) in required_confs.items():
        exp_status_dir = os.path.join(status_dir, exp_name)
        precheck_status = read_json(exp_status_dir, 'precheck.json')
        if not precheck_status['success']:
            return False, 'Config invalid for {}'.format(exp_name)

        exp_conf_dir = os.path.join(config_dir, exp_name)
        if not check_file_exists(exp_conf_dir, 'config.json'):
            return False, 'Missing config for {}'.format(exp_name)
        exp_conf = read_config(exp_conf_dir)
        if not exp_conf['active']:
            return False, 'Required experiment {} not active'.format(exp_name)
        for (entry, value) in conf_entries.items():
            if entry not in exp_conf:
                return False, 'Config for exeriment {} is missing field {}'.format(exp_name, entry)
            if not match_values(value, exp_conf[entry]):
                return False, 'Config for {} does not match requirements for entry {}'.format(exp_name, entry)

        # config is valid but need to make sure the data
        # was actually produced
        analysis_success = False
        if check_file_exists(exp_status_dir, 'analysis.json'):
            analysis_status = read_json(exp_status_dir, 'analysis.json')
            if not analysis_status['success']:
                analysis_success = False

            if not analysis_success:
                return False, '{} failed to produce analyzed data'.format(exp_name)

    return True, 'success'
