from validate_config import validate
from common import invoke_main, write_status
from summary_util import write_generic_summary

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    devs = config['devices']
    write_generic_summary(data_dir, output_dir, config['title'], devs)


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
