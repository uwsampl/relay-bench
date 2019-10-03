"""
Checks whether the required experiments have run and are
configured appropriately
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


def check_prerequisites(info, required_confs):
    """
    Takes a DashboardInfo object and a dictionary of
    exp_names -> {config field : required values}.

    Returns true if all the experiments in the required_confs
    dictionary are active, ran properly, and their configs
    contain all the fields and required values
    """
    for (exp_name, conf_entries) in required_confs.items():
        stage_statuses = info.exp_stage_statuses(exp_name)
        if not stage_statuses['precheck']['success']:
            return False, 'Config invalid for {}'.format(exp_name)

        exp_conf = info.read_exp_config(exp_name)
        if not exp_conf['active']:
            return False, 'Required experiment {} not active'.format(exp_name)
        for (entry, value) in conf_entries.items():
            if entry not in exp_conf:
                return False, 'Config for experiment {} is missing field {}'.format(exp_name, entry)
            if not match_values(value, exp_conf[entry]):
                return False, 'Config for {} does not match requirements for entry {}'.format(exp_name, entry)

        # config is valid but need to make sure the data
        # was actually produced
        if not ('analysis' in stage_statuses
                and stage_statuses['analysis']['success']):
            return False, '{} failed to produce analyzed data'.format(exp_name)

    return True, 'success'
