"""
Implementation of core dashboard infrastructure
"""
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


def attempt_parse_config(config_dir, target):
    '''
    Returns the parsed config for the target (experiment or subsystem) if it exists.
    Returns None if the config is missing or could not be parsed.
    '''
    conf_subdir = os.path.join(config_dir, target)
    if not check_file_exists(conf_subdir, 'config.json'):
        return None

    try:
        return read_json(conf_subdir, 'config.json')
    except Exception as e:
        return None


def check_present_and_executable(subdir, filenames):
    '''
    Checks that all the files in the list are present in the subdirectory
    and are executable. Returns a list of any files in the list that are
    not present or not executable.
    '''
    invalid = []
    for filename in filenames:
        path = os.path.join(subdir, filename)
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            invalid.append(filename)
    return invalid


def target_precheck(root_dir, configs_dir, target_name,
                    info_defaults, required_scripts):
    """
    Checks:
    1. That the target (subsys or experiment) config includes an 'active' field indicating whether to run it
    2. If the target is active, check that all required
    scripts are present and executable

    This function returns:
    1. a dict containing a 'status' field
    (boolean, true if all is preconfigured correctly) and a 'message' containing an
    explanation as a string if one is necessary
    2. A dict containing the target config's entries for
    each of the fields in info_defaults (uses the default
    if it's not specified)
    """
    target_conf = attempt_parse_config(configs_dir, target_name)
    if target_conf is None:
        return ({'success': False,
                 'message': 'config.json for {} is missing or fails to parse'.format(target_name)},
                None)

    update_fields = []
    target_info = {}
    for field, default in info_defaults.items():
        update_fields.append(field)
        target_info[field] = default

    for field in update_fields:
        if field in target_conf:
            target_info[field] = target_conf[field]

    # no need to check target subdirectory if it is not active
    if not target_conf['active']:
        return ({'success': True, 'message': 'Inactive'}, target_info)

    target_subdir = os.path.join(root_dir, target_name)
    if not os.path.exists(target_subdir):
        return ({'success': False,
                 'message': 'Script subdirectory for {} missing'.format(target_name)}, None)

    invalid_scripts = check_present_and_executable(target_subdir, required_scripts)
    if invalid_scripts:
        return ({
            'success': False,
            'message': 'Necessary files are missing from {} or not executable: {}'.format(
                target_subdir,
                ', '.join(invalid_scripts))
        },
                None)

    return ({'success': True, 'message': ''}, target_info)


def experiment_precheck(experiments_dir, configs_dir, exp_name):
    return target_precheck(
        experiments_dir, configs_dir, exp_name,
        {
            'active': False,
            'priority': 0,
            'rerun_setup': False,
            'tvm_remote': 'origin',
            'tvm_branch': 'master'
        },
        ['run.sh', 'analyze.sh', 'visualize.sh', 'summarize.sh'])


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


def run_all_experiments(experiments_dir, config_dir,
                        status_dir, setup_dir, data_dir,
                        graph_dir, summary_dir,
                        tmp_data_dir, data_archive,
                        time_str, randomize=True):
    """
    Handles logic for setting up and running all experiments.
    """
    exp_status = {}
    exp_confs = {}

    master_hash = get_tvm_hash()
    tvm_hashes = {}

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

    if randomize:
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


def subsystem_precheck(subsystem_dir, configs_dir, subsys_name):
    return target_precheck(
        subsystem_dir, configs_dir, subsys_name,
        {
            'active': False,
            'priority': 0
        },
        ['run.sh'])


def run_subsystem(subsystem_dir, config_dir, dashboard_home_dir,
                  output_dir, status_dir, subsys_name):
    subsys_dir = os.path.join(subsystem_dir, subsys_name)
    subsys_conf = os.path.join(config_dir, subsys_name)
    subsys_status_dir = os.path.join(status_dir, subsys_name)
    subsys_output_dir = os.path.join(output_dir, subsys_name)
    idemp_mkdir(subsys_output_dir)

    # run the run.sh file on the configs directory and the output directory
    subprocess.call([os.path.join(subsys_dir, 'run.sh'),
                     subsys_conf, dashboard_home_dir, subsys_output_dir],
                    cwd=subsys_dir)

    # collect the status file from the destination directory, copy to status dir
    status = validate_status(subsys_output_dir)
    # not literally copying because validate may have produced a status that generated an error
    write_json(subsys_status_dir, 'run.json', status)
    return status['success']


def run_all_subsystems(subsystem_dir, config_dir, dashboard_home_dir,
                       status_dir, output_dir,
                       time_str):
    """
    Handles logic for setting up and running all subsystems.
    """
    subsys_status = {}
    subsys_confs = {}

    # do the walk of subsys configs, take account of which are inactive or invalid
    for subsys_subdir, _, _ in os.walk(config_dir):
        if subsys_subdir == config_dir:
            continue
        subsys_name = os.path.basename(subsys_subdir)
        precheck, subsys_info = subsystem_precheck(subsystem_dir, config_dir, subsys_name)
        # write precheck result to status dir
        write_json(os.path.join(status_dir, subsys_name), 'precheck.json', precheck)
        subsys_status[subsys_name] = 'active'
        subsys_confs[subsys_name] = subsys_info
        if not precheck['success']:
            subsys_status[subsys_name] = 'failed'
            continue
        if not subsys_info['active']:
            subsys_status[subsys_name] = 'inactive'

    active_subsys = [subsys for subsys, status in subsys_status.items() if status == 'active']

    # high priority = go earlier, so we prioritize with negative priority, with the name as tiebreaker
    active_subsys.sort(key=lambda subsys: (-subsys_confs[subsys]['priority'], subsys))

    for subsys in active_subsys:
        success = run_subsystem(subsystem_dir, config_dir, dashboard_home_dir,
                                output_dir, status_dir, subsys)


def main(home_dir, experiments_dir, subsystem_dir):
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
    exp_config_dir = os.path.join(config_dir, 'experiments')
    subsys_config_dir = os.path.join(config_dir, 'subsystem')

    results_dir = os.path.join(home_dir, 'results')
    exp_results_dir = os.path.join(results_dir, 'experiments')
    subsys_results_dir = os.path.join(results_dir, 'subsystem')

    exp_status_dir = os.path.join(exp_results_dir, 'status')
    data_dir = os.path.join(exp_results_dir, 'data')
    graph_dir = os.path.join(exp_results_dir, 'graph')
    summary_dir = os.path.join(exp_results_dir, 'summary')

    subsys_status_dir = os.path.join(subsys_results_dir, 'status')
    subsys_output_dir = os.path.join(subsys_results_dir, 'output')

    # make a backup of the previous dashboard files if they exist
    if os.path.exists(home_dir):
        subprocess.call(['tar', '-zcf', backup_archive, home_dir])

    # directories whose contents should not change between runs of the dashboard
    persistent_dirs = {data_dir, exp_config_dir, subsys_config_dir, subsys_output_dir}
    all_dashboard_dirs = [exp_config_dir, subsys_config_dir,
                          exp_status_dir, data_dir, graph_dir, summary_dir,
                          subsys_status_dir, subsys_output_dir]

    # instantiate necessary dashboard dirs and clean any that should be empty
    for dashboard_dir in all_dashboard_dirs:
        idemp_mkdir(dashboard_dir)
        if dashboard_dir in persistent_dirs:
            continue
        for subdir, _, _ in os.walk(dashboard_dir):
            if subdir != dashboard_dir:
                subprocess.call(['rm', '-rf', subdir])

    randomize_exps = True
    if 'randomize' in dash_config:
        randomize_exps = dash_config['randomize']

    run_all_experiments(experiments_dir, exp_config_dir,
                        exp_status_dir, setup_dir, data_dir,
                        graph_dir, summary_dir,
                        tmp_data_dir, data_archive,
                        time_str, randomize=randomize_exps)

    run_all_subsystems(subsystem_dir, subsys_config_dir,
                       home_dir, subsys_status_dir,
                       subsys_output_dir, time_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read dashboard config')
    parser.add_argument('--home-dir', type=str, required=True)
    parser.add_argument('--experiments-dir', type=str, required=True)
    parser.add_argument('--subsystem-dir', type=str, required=True)
    args = parser.parse_args()
    main(args.home_dir, args.experiments_dir, args.subsystem_dir)
