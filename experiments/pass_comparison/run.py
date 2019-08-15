import argparse

from validate_config import validate
from common import write_status
from trial_util import run_trials
from relay_util import cnn_setup, cnn_trial, cnn_teardown

def passes_setup(network, dev, batch_size, pass_spec):
    members = pass_spec.split(';')
    opt_level = int(members[0])
    pass_list = members[1]

    # baseline: don't specify any passes
    if opt_level == 0 and pass_list == '':
        return cnn_setup(network, dev, batch_size, opt_level,
                         use_passes=False)

    return cnn_setup(network, dev, batch_size, opt_level,
                     use_passes=True, passes=pass_list)


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    # We must preprocess the passes to work with passes_setup.
    # I.e., we must serialize it so it can be written to CSV,
    # so we separate the pass list by |'s and the opt_level with
    # a semicolon
    passes = [';'.join([str(pass_spec[0]), '|'.join(pass_spec[1])])
              for pass_spec in config['passes']]

    success, msg = run_trials(
        'relay', 'pass_comparison',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        cnn_trial, passes_setup, cnn_teardown,
        ['network', 'device', 'batch_size', 'pass_spec'],
        [config['networks'], config['devices'],
         config['batch_sizes'], passes],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir)
