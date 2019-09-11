import sys

from validate_config import validate
from common import invoke_main, write_status
from trial_util import run_trials, configure_seed
from relay_util import cnn_setup, cnn_trial, cnn_teardown

def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)
        return

    if 'relay' not in config['frameworks']:
        write_status(output_dir, True, 'Relay not run')
        sys.exit(0)

    configure_seed(config)

    opt_levels = [config['relay_opt']]

    success, msg = run_trials(
        'relay', 'cnn_comp',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        cnn_trial, cnn_setup, cnn_teardown,
        ['network', 'device', 'batch_size', 'opt_level'],
        [config['networks'], config['devices'],
         config['batch_sizes'], opt_levels],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'output_dir')
