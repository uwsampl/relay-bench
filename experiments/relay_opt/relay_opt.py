from validate_config import validate
from common import invoke_main, write_status
from trial_util import run_trials, configure_seed
from relay_util import cnn_setup, cnn_trial, cnn_teardown

def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    configure_seed(config)

    success, msg = run_trials(
        'relay', 'opt_comparison',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        cnn_trial, cnn_setup, cnn_teardown,
        ['network', 'device', 'batch_size', 'opt_level'],
        [config['networks'], config['devices'],
         config['batch_sizes'], config['opt_levels']],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'output_dir')
