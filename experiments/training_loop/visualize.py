from common import invoke_main, write_status


def main(data_dir, config_dir, output_dir):
    # TODO: fill in
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
