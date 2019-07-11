'''Implementation of core dashboard infrastructure'''
import argparse
import datetime
import json
import os
import sys
import subprocess

def check_file_exists(dirname, filename):
    full_name = os.path.join(data_prefix, filename)
    return os.isfile(full_name)


def prepare_out_file(dirname, filename):
    full_name = os.path.join(dirname, filename)
    if not check_file_exists(dirname, filename):
        os.makedirs(os.path.dirname(full_name))
    return full_name


def read_json(dirname, filename):
    with open(os.path.join(dirname, filename)) as json_file:
        data = json.load(json_file)
        return data


def write_json(dirname, filename, obj):
    filename = prepare_out_file(dirname, filename)
    with open(filename, 'w') as outfile:
        json.dump(obj, outfile)


def validate_status(dirname):
    if not check_file_exists(dirname, 'status.json'):
        return {'success': False, 'message': 'No status.json in {}'.format(dirname)}
    status = read_json(dirname, 'status.json')
    if 'success' not in status:
        return {'success': False,
                'message': 'status.json in {} has no \'success\' field'.format(dirname)}
    if 'message' not in status:
        return {'success': False,
                'message': 'status.json in {} has no \'message\' field'.format(dirname)}
    return status


def experiment_precheck(experiments_dir, configs_dir, exp_name):
    """
    Checks:
    1. That the experiment config includes an 'active' field indicating whether to
    run the experiment
    2. If the experiment is active, check that all required are present in the experiment
    directory for the given experiment

    This function returns:
    1. a dict containing a 'status' field
    (boolean, true if all is preconfigured correctly) and a 'message' containing an
    explanation as a string if one is necessary
    2. whether the experiment is active (boolean), False if experiment is invalid
    """
    conf_subdir = os.path.join(configs_dir, exp_name)
    if not os.path.exists(conf_subdir):
        return ({'success': False,
                 'message': 'Config directory for experiment {} is missing'.format(exp_name)},
                False)
    conf_file = os.path.join(conf_subdir, 'config.json')
    if not os.path.isfile(path):
        return ({'success': False,
                 'message': 'config.json for experiment {} is missing'.format(exp_name)},
                False)
    exp_conf = read_json(conf_subdir, 'config.json')
    if 'active' not in exp_conf:
        return ({'success': False,
                 'message': 'config.json for experiment {} has no active field'.format(exp_name)},
                False)

    # no need to check experiment subdirectory if the experiment itself is not active
    if not exp_conf['active']:
        return ({'success': True, 'message': 'Inactive'}, False)

    exp_subdir = os.path.join(experiments_dir, exp_name)
    if not os.path.exists(exp_subdir):
        return ({'success': False,
                 'message': 'Experiment subdirectory {} missing'.format(exp_name)}, False)
    required_scripts = ['run.sh', 'analyze.sh', 'visualize.sh', 'summarize.sh']
    for script in required_scripts:
        path = os.path.join(exp_subdir, script)
        if not os.path.isfile(path):
            return ({'success': False,
                     'message': 'Required file {} is missing from {}'.format(script, exp_subdir)},
                    False)
        if not os.access(path, os.X_OK):
            return ({'success': False,
                     'message': 'Required file {} in {} is not executable'.format(script,
                                                                                 exp_subdir)},
                    False)
    return ({'success': True, 'message': ''}, True)


def run_experiment(experiments_dir, configs_dir, tmp_data_dir, status_dir, exp_name):
    # set up a temporary data directory for that experiment
    exp_data_dir = os.path.join(tmp_data_dir, exp_name)
    os.makedirs(exp_data_dir)

    # run the run.sh file on the configs directory and the destination directory
    subprocess.call([os.path.join(experiments_dir, exp_name, 'run.sh'),
                     os.path.join(configs_dir, exp_name),
                     exp_data_dir])

    # collect the status file from the destination directory, copy to status dir
    status = validate_status(exp_data_dir)
    # not literally copying because validate may have produced a status that generated an error
    write_json(os.path.join(status_dir, exp_name), 'run.json', status)
    return status['success']


def analyze_experiment(experiments_dir, tmp_data_dir, data_dir, status_dir, date_str, exp_name):
    exp_data_dir = os.path.join(tmp_data_dir, exp_name)
    tmp_analysis_dir = os.path.join(exp_data_dir, 'analysis')
    os.makedirs(tmp_analysis_dir)

    analyzed_data_dir = os.path.join(data_dir, exp_name)
    if not os.path.exists(analyzed_data_dir):
        os.makedirs(analyzed_data_dir)

    subprocess.call([os.path.join(experiments_dir, exp_name, 'analyze.sh'),
                     exp_data_dir, tmp_analysis_dir])

    status = validate_status(tmp_analysis_dir)

    # read the analyzed data, append a timestamp field, and copy over to the permanent data dir
    if status['success']:
        data_exists = check_file_exists(tmp_analysis_dir, 'data.json')
        if not data_exists:
            status = {'success': False, 'message': 'No data.json file produced by {}'.format(exp_name)}
        else:
            data = read_json(tmp_analysis_dir, 'data.json')
            data['timestamp'] = date_str
            write_json(analyzed_data_dir, 'data_{}.json'.format(date_str), data)

    write_json(os.path.join(status_dir, exp_name), 'analysis.json', status)
    return status['success']


def visualize_experiment(experiments_dir, data_dir, graph_dir, status_dir, exp_name):
    exp_graph_dir = os.path.join(graph_dir, exp_name)
    exp_data_dir = os.path.join(data_dir, exp_name)
    subprocess.call([os.path.join(experiments_dir, exp_name, 'visualize.sh'),
                     exp_data_dir, exp_graph_dir])

    status = validate_status(exp_graph_dir)
    write_json(os.path.join(status_dir, exp_name), 'visualization.json', status)


def summary_valid(exp_summary_dir):
    """
    Checks that the experiment summary directory contains a summary.json
    file and that the summary.json file contains the required fields, title
    and value.
    """
    exists = check_file_exists(exp_summary_dir, 'summary.json')
    if not exists:
        return False
    summary = read_json(exp_summary_dir, 'summary.json')
    return 'title' in summary and 'value' in summary


def summarize_experiment(experiments_dir, data_dir, summary_dir, status_dir, exp_name):
    exp_summary_dir = os.path.join(summary_dir, exp_name)
    exp_data_dir = os.path.join(data_dir, exp_name)
    subprocess.call([os.path.join(experiments_dir, exp_name, 'summarize.sh'),
                     exp_data_dir, exp_summary_dir])

    status = validate_status(exp_summary_dir)
    if status['success'] and not summary_valid(exp_summary_dir):
        status = {
            'success': False,
            'message': 'summary.json produced by {} is invalid'.format(exp_name)
        }
    write_json(os.path.join(status_dir, exp_name), 'summary.json', status)


def main(home_dir, experiments_dir):
    """
    Home directory: Where config info for experiments, etc., is
    Experiments directory: Where experiment implementations are
    Both should be given as absolute directories
    """
    time_of_run = datetime.datetime.now()
    time_str = time_of_run.strftime('%m-%d-%Y-%H%M')

    if not check_file_exists(home_dir, 'config.json'):
        print('Dashboard config (config.json) is missing in {}'.format(home_dir))
        sys.exit(1)
    dash_config = read_json(home_dir, 'config.json')

    tmp_data_dir = os.path.join(dash_config['tmp_data_dir'], 'benchmarks_' + time_str)
    data_archive = os.path.join(dash_config['tmp_data_dir'], 'benchmarks_' + time_str + '_data.tar.gz')
    backup_archive = os.path.join(dash_config['backup_dir'], 'dashboard_' + time_str + '.tar.gz')

    config_dir = os.path.join(home_dir, 'config')
    status_dir = os.path.join(home_dir, 'status')
    data_dir = os.path.join(home_dir, 'data')
    graph_dir = os.path.join(home_dir, 'graph')
    summary_dir = os.path.join(home_dir, 'summary')

    # make a backup of the previous dashboard files if they exist
    if os.path.exists(home_dir):
        subprocess.call(['tar', '-zcf', backup_archive, home_dir])
    for dashboard_dir in [config_dir, status_dir, data_dir, graph_dir, summary_dir]:
        if not os.path.exists(dashboard_dir):
            os.makedirs(dashboard_dir)
            continue
        # remove subdirectories to set up for new run (except data, config)
        if dashboard_dir == data_dir or dashboard_dir == config_dir:
            continue
        for subdir, _, _ in os.walk(dashboard_dir):
            if subdir != dashboard_dir:
                subprocess.call(['rm', '-rf', subdir])

    exp_status = {}

    # do the walk of experiment configs, take account of which experiments are
    # either inactive or invalid
    for conf_subdir, _, _ in os.walk(config_dir):
        if conf_subdir == config_dir:
            continue
        exp_name = os.path.basename(conf_subdir)
        precheck, active = experiment_precheck(experiments_dir, config_dir, exp_name)
        # write precheck result to status dir
        write_json(status_dir, 'precheck.json', precheck)
        exp_status[name] = 'active'
        if not precheck['success']:
            exp_status[name] = 'failed'
            continue
        if not active:
            exp_status[name] = 'inactive'

    # (run first because if anything goes wrong later, we can use the raw data)
    # for each active experiment, run and generate data
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']
    for exp in active_exps:
        success = run_experiment(experiments_dir, config_dir, tmp_data_dir, status_dir, exp)
        if not success:
            exp_status[exp] = 'failed'

    # for each active experiment not yet eliminated, run analysis
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']
    for exp in active_exps:
        success = analyze_experiment(experiments_dir, tmp_data_dir, data_dir, status_dir,
                                     time_str, exp)
        if not success:
            exp_status[exp] = 'failed'

    # after analysis we can compress the data
    subprocess.call(['tar', '-zcf', data_archive, tmp_data_dir])
    subprocess.call(['rm', '-rf', tmp_data_dir])

    # for each experiment for which analysis succeeded, run visualization and summarizaion
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']
    for exp in active_exps:
        visualize_experiment(experiments_dir, data_dir, graph_dir, status_dir, exp)
        summarize_experiment(experiments_dir, data_dir, summary_dir, status_dir, exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read dashboard config')
    parser.add_argument('--home-dir', type=str, required=True)
    parser.add_argument('--experiments-dir', type=str, required=True)
    args = parser.parse_args()
    main(args.home_dir, args.experiments_dir)
