"""
Functions encapsulating the most common structure
for dashboard steps. Experiments are free to deviate
from these; these have simply arisen from the most
common cases in practice.
"""
import os

from collections import OrderedDict
from collections.abc import Iterable

from common import (invoke_main, write_status,
                    sort_data, time_difference,
                    render_exception, write_json)
from trial_util import run_trials, configure_seed
from analysis_util import trials_stat_summary, add_detailed_summary
from summary_util import write_generic_summary
from plot_util import (generate_longitudinal_comparisons,
                       PlotBuilder, PlotScale, PlotType)


def common_trial_params(fw, exp_name, trial_func, trial_setup, trial_teardown,
                        fieldnames, conf_keys):
    """
    Returns a function that takes an experiment config and
    returns a list of parameters to trial_util.run_trials
    that corresponds to the most common experiment setups.

    Parameters
    ==========
    fw: str, Framework name
    exp_name: str, Name of the experiment
    trial_func, trial_setup, trial_teardown: see trial_util.run_trials
    fieldnames: [str], list of experiment output fields
    conf_keys: [str], list of dictionary keys in the config
                      corresponding to fieldnames (in order)
    """
    def gen_trial_params(config):
        config_values = []
        for key in conf_keys:
            key_values = config[key]
            if not isinstance(key_values, Iterable):
                key_values = [key_values]
            config_values.append(key_values)

        return [
            fw, exp_name,
            config['dry_run'], config['n_times_per_input'], config['n_inputs'],
            trial_func, trial_setup, trial_teardown,
            fieldnames, config_values
        ]
    return gen_trial_params


def common_early_exit(field_contains):
    """
    Returns an 'early exit' function that takes a config dictionary
    and checks that for every field (key) in field_contains,
    the specified value is in the corresponding config field
    (which is assumed to be a list or set)
    """
    def early_exit(config):
        for field, value in field_contains.items():
            if value not in config[field]:
                return True, '{} not in config {}'.format(value, field)
        return False, ''
    return early_exit


def run_template(validate_config, check_early_exit=None, gen_trial_params=None):
    """
    Common template for the "run" step of an experiment.
    Reads a config directory and output directory from the command
    line, reads in the experiment config, and uses it to generate
    parameters for trial_util.run_trials.

    Exits with a ret code of 1 if there is any problem or exception,
    otherwise exits with 0

    Parameters
    ==========
    validate_config : A function from string-keyed dictionary
        to (dictionary, str). If this function is specified, it
        will be run on the object parsed from the experiment's
        config.json. Returns a processed config if successful,
        or None and an error message if there is a problem
        with the config

    check_early_exit: Function from config -> (bool, str). If
        specified, this function will be run on the experiment
        config and determine whether the script should exit without
        running the experiment. The second return value should be a message explaining
        why the function exited early (if the first return value is true).

    gen_trial_params: Function that takes a config and returns
         an array of arguments to trial_util.run_trial.
         If this is omitted, no experiment will run.
    """
    def main(config_dir, output_dir):
        try:
            config, msg = validate_config(config_dir)
            if config is None:
                write_status(output_dir, False, msg)
                return 1

            if check_early_exit is not None:
                early_exit, msg = check_early_exit(config)
                if early_exit:
                    write_status(output_dir, True, msg)
                    return 0

            configure_seed(config)

            if gen_trial_params is None:
                write_status(output_dir, True, 'No trial to run')
                return 0

            trial_params = gen_trial_params(config)
            success, msg = run_trials(*trial_params, path_prefix=output_dir)
            write_status(output_dir, success, msg)
            return 0 if success else 1
        except Exception as e:
            write_status(output_dir, False, render_exception(e))
            return 1

    invoke_main(main, 'config_dir', 'output_dir')


def generate_graphs_by_dev(visualize):
    """
    Returns a function that takes a config, raw data object,
    and output directory and invokes
    visualize(dev, raw_data[dev], output_dir/comparison)
    """
    def generate(config, raw_data, output_dir):
        devs = config['devices']
        comparison_dir = os.path.join(output_dir, 'comparison')
        for dev in devs:
            visualize(dev, raw_data[dev], comparison_dir)
    return generate


def common_individual_comparison(x_name, title_stem, filename_stem,
                                 scale=PlotScale.LOG,
                                 use_networks=True,
                                 cnn_name_map=True):
    """
    Returns a function that creates a graph of per-model
    performance on each device
    """
    model_to_text = {
        'nature-dqn': 'Nature DQN',
        'vgg-16': 'VGG-16',
        'resnet-18': 'ResNet-18',
        'mobilenet': 'MobileNet'
    }

    def visualize(dev, raw_data, output_dir):
        # empty data: nothing to do
        if not raw_data.items():
            return

        plot_type = PlotType.MULTI_BAR if use_networks else PlotType.BAR

        data_copy = dict(raw_data)

        if use_networks and cnn_name_map:
            renamed_data = {}
            # make model names presentable
            for (dev, models) in data_copy.items():
                renamed_data[dev] = {}
                for model in models.keys():
                    val = models[model]
                    renamed_data[dev][model_to_text[model]] = val
            data_copy = renamed_data

        sorted_raw = OrderedDict(sorted(data_copy.items()))
        data = {
            'raw': sorted_raw,
            'meta': [x_name, 'Mean Inference Time (ms)']
        }
        if use_networks:
            data = {
                'raw': sorted_raw,
                'meta': [x_name, 'Network', 'Mean Inference Time (ms)']
            }

        builder = PlotBuilder() \
                  .set_title('{} on {}'.format(
                      title_stem, dev.upper())) \
                  .set_x_label(x_name) \
                  .set_y_label('Mean Inference Time (ms)') \
                  .set_y_scale(scale)

        # TODO(@weberlo): this results in a bug in the non-network cases
        # for some reason
        if use_networks:
            builder.set_figure_height(3.0) \
                   .set_aspect_ratio(3.3) \
                   .set_sig_figs(3)

        builder.make(plot_type, data) \
               .save(output_dir, '{}-{}.png'.format(
                   filename_stem, dev))

    return generate_graphs_by_dev(visualize)


def visualize_template(validate_config, generate_individual_comparisons):
    """
    Common template for the "visualize" step of an experiment.

    Reads data, config, output directories from the command
    line, reads in the experiment config, and all the data
    in the data directory.

    Runs generate_individual_comparisons on the most recent
    data file with the config. Also generates lognitudinal
    comparisons (using the basic function) over all time
    and over the last two weeks.

    Exits with a ret code of 1 if there is any problem or exception,
    otherwise exits with 0

    Parameters
    ==========
    validate_config : A function from string-keyed dictionary
        to (dictionary, str). If this function is specified, it
        will be run on the object parsed from the experiment's
        config.json. Returns a processed config if successful,
        or None and an error message if there is a problem
        with the config

    generate_individual_comparisons : A function that, given
        a valid parsed config, the most recent data object,
        and an output directory, produces graphs of the given data
    """
    def main(data_dir, config_dir, output_dir):
        try:
            config, msg = validate_config(config_dir)
            if config is None:
                write_status(output_dir, False, msg)
                return 1

            all_data = sort_data(data_dir)
            most_recent = all_data[-1]
            last_two_weeks = [entry for entry in all_data
                              if time_difference(most_recent, entry).days < 14]

            generate_individual_comparisons(config, most_recent, output_dir)
            generate_longitudinal_comparisons(all_data, output_dir, 'all_time')
            generate_longitudinal_comparisons(last_two_weeks, output_dir, 'two_weeks')
        except Exception as e:
            write_status(output_dir, False,
                         'Exception encountered:\n' + render_exception(e))
            return 1

        write_status(output_dir, True, 'success')

    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')


def summarize_template(validate_config, use_networks=True):
    """
    Common template for the "visualize" step of an experiment.

    Reads data, config, output directories from the command
    line, reads in the experiment config.

    Uses write_generic_summary to produce a summary based on
    the most recent data file based on the devices and title
    specified in the config.

    Exits with a ret code of 1 if there is any problem or exception,
    otherwise exits with 0

    Parameters
    ==========
    validate_config : A function from string-keyed dictionary
        to (dictionary, str). If this function is specified, it
        will be run on the object parsed from the experiment's
        config.json. Returns a processed config if successful,
        or None and an error message if there is a problem
        with the config

    use_networks : Whether to summarize by networks (true by default)
    """
    def main(data_dir, config_dir, output_dir):
        config, msg = validate_config(config_dir)
        if config is None:
            write_status(output_dir, False, msg)
            return 1

        devs = config['devices']
        networks = []
        if use_networks:
            networks = config['networks']
        write_generic_summary(data_dir, output_dir, config['title'],
                              devs, networks, use_networks=use_networks)

    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')


def analysis_template(validate_config, generate_listing_settings,
                      generate_data_query, use_networks=True):
    """
    Common template for the "visualize" step of an experiment.

    Reads data, config, output directories from the command
    line, reads in the experiment config.

    Uses trials_stat_summary and the user-specified functions
    to query the raw data and produce a data.json file

    Exits with a ret code of 1 if there is any problem or exception,
    otherwise exits with 0

    Parameters
    ==========
    validate_config : A function from string-keyed dictionary
        to (dictionary, str). If this function is specified, it
        will be run on the object parsed from the experiment's
        config.json. Returns a processed config if successful,
        or None and an error message if there is a problem
        with the config

    generate_listing_settings: A function that takes the exp
        config and generates a mapping of category names to
        all information that will be needed to generate a data
        query (args to trials_stat_summary) for each category's
        corresponding data

    generate_data_query: A function that takes the config, a device,
        network (if use_networks is True), and a listing setting
        and returns a set of query arguments for trials_stat_summary

    use_networks: Assumes the config has multiple networks and
        the analysis should analyze each network separately.
        True by default
    """
    def main(data_dir, config_dir, output_dir):
        config, msg = validate_config(config_dir)
        if config is None:
            write_status(output_dir, False, msg)
            return 1

        devs = config['devices']
        listing_settings = generate_listing_settings(config)

        ret = {}
        for dev in devs:
            ret[dev] = {}
            for listing, settings in listing_settings.items():
                if not use_networks:
                    query = generate_data_query(config, dev, settings)
                    summary, success, msg = trials_stat_summary(data_dir, *query)
                    if not success:
                        write_status(output_dir, False, msg)
                        return 1

                    ret[dev][listing] = summary['mean']
                    add_detailed_summary(ret, summary, dev, listing)
                    continue

                ret[dev][listing] = {}
                for network in config['networks']:
                    query = generate_data_query(config, dev, network, settings)
                    summary, success, msg = trials_stat_summary(data_dir, *query)
                    if not success:
                        write_status(output_dir, False, msg)
                        return 1

                    ret[dev][listing][network] = summary['mean']
                    add_detailed_summary(ret, summary, dev, listing, network)
        write_json(output_dir, 'data.json', ret)
        write_status(output_dir, True, 'success')

    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
