import argparse

from validate_config import validate
from common import write_status
from trial_util import run_trials
from relay_util import cnn_setup, cnn_trial, cnn_teardown

def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    # must preprocess the passes to work with cnn_setup
    # (has to be a |-separated list in order to be written to CSV)
    passes = ['|'.join(pass_list) for pass_list in config['passes']]

    success, msg = run_trials(
        'relay', 'pass_comparison',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        cnn_trial, cnn_setup, cnn_teardown,
        ['network', 'device', 'batch_size', 'opt_level', 'pass'],
        [config['networks'], config['devices'],
         config['batch_sizes'], [0], passes],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir)
