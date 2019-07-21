import argparse
import os

from validate_config import validate
from common import write_status, read_json, write_json
from analysis_util import trials_average_time

def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    # No further analysis is required beyond the raw stats reported by the VTA
    # simulator, so we just propagate the data to the next stage of the
    # pipeline.
    data = read_json(data_dir, 'data.json')
    write_json(output_dir, 'data.json', data)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.config_dir, args.output_dir)
