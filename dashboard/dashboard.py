'''Implementation of core dashboard infrastructure'''
import argparse
import datetime
import json
import os
import random
import sys
import subprocess
import time

from common import (check_file_exists, idemp_mkdir, prepare_out_file,
                    read_json, write_json, read_config)

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


def build_tvm_branch(remote, branch):
    """
    Given a remote URL and a branch, this function sets the TVM install
    to that branch and rebuilds.
    """
    # ugly to call a bash script like this but better than specifying
    # all the git commands in Python
    benchmark_deps = os.environ['BENCHMARK_DEPS']
    bash_deps = os.path.join(benchmark_deps, 'bash')
    subprocess.call([os.path.join(bash_deps, 'build_tvm_branch.sh'),
                     remote, branch],
                    cwd=bash_deps)


def get_tvm_hash():
    tvm_home = os.environ['TVM_HOME']
    git_check = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                        cwd=tvm_home)
    return git_check.decode('UTF-8').strip()


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
    2. A dict with the following dashboard-relevant config fields, None if there's an error:
      {
        'active': (boolean, whether the experiment is active)
        'priority': (int, experiment's priority in the ordering)
        'tvm_remote': (string, TVM remote repo to pull from, 'origin' if not specified)
        'tvm_branch': (string, TVM branch to check out, 'master' if not specified)
      }
    """
    conf_subdir = os.path.join(configs_dir, exp_name)
    if not os.path.exists(conf_subdir):
        return ({'success': False,
                 'message': 'Config directory for experiment {} is missing'.format(exp_name)},
                None)
    conf_file = os.path.join(conf_subdir, 'config.json')
    if not os.path.isfile(conf_file):
        return ({'success': False,
                 'message': 'config.json for experiment {} is missing'.format(exp_name)},
                None)

    exp_conf = None
    try:
        exp_conf = read_json(conf_subdir, 'config.json')
    except Exception as e:
        return ({'success': False,
                 'message': 'Exception when parsing config.json for experiment {}'.format(exp_name)},
                None)

    if 'active' not in exp_conf:
        return ({'success': False,
                 'message': 'config.json for experiment {} has no active field'.format(exp_name)},
                 None)

    exp_info = {
        'active': False,
        'priority': 0,
        'rerun_setup': False,
        'tvm_remote': 'origin',
        'tvm_branch': 'master'
    }
    update_fields = ['active', 'tvm_remote', 'tvm_branch', 'priority']
    for field in update_fields:
        if field in exp_conf:
            exp_info[field] = exp_conf[field]

    # no need to check experiment subdirectory if the experiment itself is not active
    if not exp_conf['active']:
        return ({'success': True, 'message': 'Inactive'}, exp_info)

    exp_subdir = os.path.join(experiments_dir, exp_name)
    if not os.path.exists(exp_subdir):
        return ({'success': False,
                 'message': 'Experiment subdirectory {} missing'.format(exp_name)}, None)
    required_scripts = ['run.sh', 'analyze.sh', 'visualize.sh', 'summarize.sh']
    for script in required_scripts:
        path = os.path.join(exp_subdir, script)
        if not os.path.isfile(path):
            return ({'success': False,
                     'message': 'Required file {} is missing from {}'.format(script, exp_subdir)},
                    None)
        if not os.access(path, os.X_OK):
            return ({'success': False,
                     'message': 'Required file {} in {} is not executable'.format(script,
                                                                                 exp_subdir)},
                    None)

    return ({'success': True, 'message': ''}, exp_info)


def has_setup(experiments_dir, exp_name):
    setup_path = os.path.join(experiments_dir, exp_name, 'setup.sh')
    return os.path.isfile(setup_path) and os.access(setup_path, os.X_OK)


def most_recent_experiment_update(experiments_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)
    git_list = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=exp_dir)
    files = git_list.decode('UTF-8').strip().split('\n')
    most_recent = None
    for f in files:
        raw_date = subprocess.check_output(['git', 'log', '-1', '--format=\"%ad\"', '--', f], cwd=exp_dir)
        date_str = raw_date.decode('UTF-8').strip(' \"\n')
        parsed = datetime.datetime.strptime(date_str, '%a %b %d %H:%M:%S %Y %z')
        if most_recent is None or most_recent < parsed:
            most_recent = parsed
    return most_recent


def last_setup_time(setup_dir, exp_name):
    marker_file = os.path.join(setup_dir, exp_name, '.last_setup')
    if os.path.isfile(marker_file):
        t = os.path.getmtime(marker_file)
        return time.localtime(t)
    return None


def should_setup(experiments_dir, setup_dir, exp_name):
    last_setup = last_setup_time(setup_dir, exp_name)
    if last_setup is None:
        return True

    most_recent = most_recent_experiment_update(experiments_dir, exp_name).timetuple()
    return most_recent > last_setup


def setup_experiment(experiments_dir, configs_dir, setup_dir, status_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)

    exp_conf = os.path.join(configs_dir, exp_name)
    exp_setup_dir = os.path.join(setup_dir, exp_name)
    exp_status_dir = os.path.join(status_dir, exp_name)

    # remove the existing setup dir before running the script again
    subprocess.call(['rm', '-rf', exp_setup_dir])
    idemp_mkdir(exp_setup_dir)

    subprocess.call([os.path.join(exp_dir, 'setup.sh'), exp_conf, exp_setup_dir], cwd=exp_dir)

    status = validate_status(exp_setup_dir)
    write_json(exp_status_dir, 'setup.json', status)

    # if setup succeeded, touch a marker file so we know what time to check for changes
    if status['success']:
        subprocess.call(['touch', '.last_setup'], cwd=exp_setup_dir)

    return status['success']


def copy_setup(experiments_dir, setup_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)
    exp_setup_dir = os.path.join(setup_dir, exp_name)
    subprocess.call(['cp', '-r', os.path.join(exp_setup_dir, '.'), 'setup/'],
                    cwd=exp_dir)


def run_experiment(experiments_dir, configs_dir, tmp_data_dir, status_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)
    exp_conf = os.path.join(configs_dir, exp_name)

    # set up a temporary data directory for that experiment
    exp_data_dir = os.path.join(tmp_data_dir, exp_name)
    exp_status_dir = os.path.join(status_dir, exp_name)
    idemp_mkdir(exp_data_dir)

    # run the run.sh file on the configs directory and the destination directory
    subprocess.call([os.path.join(exp_dir, 'run.sh'), exp_conf, exp_data_dir],
                    cwd=exp_dir)

    # collect the status file from the destination directory, copy to status dir
    status = validate_status(exp_data_dir)
    # not literally copying because validate may have produced a status that generated an error
    write_json(exp_status_dir, 'run.json', status)
    return status['success']


def analyze_experiment(experiments_dir, configs_dir, tmp_data_dir,
                       data_dir, status_dir, date_str, tvm_hash, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)

    exp_data_dir = os.path.join(tmp_data_dir, exp_name)
    exp_config_dir = os.path.join(configs_dir, exp_name)
    tmp_analysis_dir = os.path.join(exp_data_dir, 'analysis')
    idemp_mkdir(tmp_analysis_dir)

    analyzed_data_dir = os.path.join(data_dir, exp_name)
    if not os.path.exists(analyzed_data_dir):
        idemp_mkdir(analyzed_data_dir)

    subprocess.call([os.path.join(exp_dir, 'analyze.sh'),
                     exp_config_dir, exp_data_dir, tmp_analysis_dir],
                    cwd=exp_dir)

    status = validate_status(tmp_analysis_dir)

    # read the analyzed data, append a timestamp field, and copy over to the permanent data dir
    if status['success']:
        data_exists = check_file_exists(tmp_analysis_dir, 'data.json')
        if not data_exists:
            status = {'success': False, 'message': 'No data.json file produced by {}'.format(exp_name)}
        else:
            data = read_json(tmp_analysis_dir, 'data.json')
            data['timestamp'] = date_str
            data['tvm_hash'] = tvm_hash
            write_json(analyzed_data_dir, 'data_{}.json'.format(date_str), data)

    write_json(os.path.join(status_dir, exp_name), 'analysis.json', status)
    return status['success']


def visualize_experiment(experiments_dir, configs_dir, data_dir,
                         graph_dir, status_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)

    exp_graph_dir = os.path.join(graph_dir, exp_name)
    exp_config_dir = os.path.join(configs_dir, exp_name)
    exp_data_dir = os.path.join(data_dir, exp_name)
    subprocess.call([os.path.join(exp_dir, 'visualize.sh'),
                     exp_config_dir, exp_data_dir, exp_graph_dir],
                    cwd=exp_dir)

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


def summarize_experiment(experiments_dir, configs_dir, data_dir,
                         summary_dir, status_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)

    exp_summary_dir = os.path.join(summary_dir, exp_name)
    exp_config_dir = os.path.join(configs_dir, exp_name)
    exp_data_dir = os.path.join(data_dir, exp_name)
    subprocess.call([os.path.join(exp_dir, 'summarize.sh'),
                     exp_config_dir, exp_data_dir, exp_summary_dir],
                    cwd=exp_dir)

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

    exp_confs = {}

    master_hash = get_tvm_hash()
    tvm_hashes = {}

    if not check_file_exists(home_dir, 'config.json'):
        print('Dashboard config (config.json) is missing in {}'.format(home_dir))
        sys.exit(1)
    dash_config = read_json(home_dir, 'config.json')

    # must expand all tildes in the config to avoid future errors
    for path_field in ['tmp_data_dir', 'setup_dir', 'backup_dir']:
        dash_config[path_field] = os.path.expanduser(dash_config[path_field])

    tmp_data_dir = os.path.join(dash_config['tmp_data_dir'], 'benchmarks_' + time_str)
    data_archive = os.path.join(dash_config['tmp_data_dir'], 'benchmarks_' + time_str + '_data.tar.gz')
    setup_dir = dash_config['setup_dir']
    backup_archive = os.path.join(dash_config['backup_dir'], 'dashboard_' + time_str + '.tar.gz')
    idemp_mkdir(tmp_data_dir)
    idemp_mkdir(os.path.dirname(backup_archive))
    idemp_mkdir(setup_dir)

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
            idemp_mkdir(dashboard_dir)
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
        precheck, exp_info = experiment_precheck(experiments_dir,
                                                 config_dir,
                                                 exp_name)
        # write precheck result to status dir
        write_json(os.path.join(status_dir, exp_name), 'precheck.json', precheck)
        exp_status[exp_name] = 'active'
        exp_confs[exp_name] = exp_info
        if not precheck['success']:
            exp_status[exp_name] = 'failed'
            continue
        if not exp_info['active']:
            exp_status[exp_name] = 'inactive'

    active_exps = [exp for exp, status in exp_status.items() if status == 'active']

    # handle setup for all experiments that have it
    for exp in active_exps:
        if has_setup(experiments_dir, exp):
            # run setup if the most recent updated file is more recent
            # than the last setup run or if the flag to rerun is set
            if should_setup(experiments_dir, setup_dir, exp) or exp_confs[exp]['rerun_setup']:
                success = setup_experiment(experiments_dir, config_dir, setup_dir, status_dir, exp)
                if not success:
                    exp_status[exp_name] = 'failed'
                    continue
            # copy over the setup files regardless of whether we ran it this time
            copy_setup(experiments_dir, setup_dir, exp)

    # for each active experiment, run and generate data
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']

    randomize_exps = True
    if 'randomize' in dash_config:
        randomize_exps = dash_config['randomize']

    if randomize_exps:
        random.shuffle(active_exps)
    else:
        # if experiment order is not random, sort by experiment priority,
        # with name as a tie-breaker. Since we want higher priority exps to
        # be first, we use -priority as the first element of the key
        active_exps.sort(key=lambda exp: (-exp_confs[exp]['priority'], exp))

    for exp in active_exps:
        # handle TVM branching if the experiment needs a different TVM
        (remote, branch) = (exp_confs[exp]['tvm_remote'], exp_confs[exp]['tvm_branch'])
        used_branch = False
        tvm_hash = master_hash

        if remote != 'origin' or branch != 'master':
            used_branch = True
            build_tvm_branch(remote, branch)
            tvm_hash = get_tvm_hash()

        tvm_hashes[exp] = tvm_hash

        success = run_experiment(experiments_dir, config_dir, tmp_data_dir, status_dir, exp)
        if not success:
            exp_status[exp] = 'failed'

        if used_branch:
            build_tvm_branch('origin', 'master')

    # for each active experiment not yet eliminated, run analysis
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']
    for exp in active_exps:
        success = analyze_experiment(experiments_dir, config_dir, tmp_data_dir, data_dir,
                                     status_dir, time_str, tvm_hashes[exp], exp)
        if not success:
            exp_status[exp] = 'failed'

    # after analysis we can compress the data
    subprocess.call(['tar', '-zcf', data_archive, tmp_data_dir])
    subprocess.call(['rm', '-rf', tmp_data_dir])

    # for each experiment for which analysis succeeded, run visualization and summarization
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']
    for exp in active_exps:
        visualize_experiment(experiments_dir, config_dir, data_dir, graph_dir, status_dir, exp)
        summarize_experiment(experiments_dir, config_dir, data_dir, summary_dir, status_dir, exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read dashboard config')
    parser.add_argument('--home-dir', type=str, required=True)
    parser.add_argument('--experiments-dir', type=str, required=True)
    args = parser.parse_args()
    main(args.home_dir, args.experiments_dir)
