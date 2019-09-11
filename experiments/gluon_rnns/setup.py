import argparse
from common import render_exception, write_status
from mxnet_util import export_mxnet_model

def main(config_dir, setup_dir):
    try:
        export_mxnet_model('rnn', setup_dir)
        export_mxnet_model('gru', setup_dir)
        export_mxnet_model('lstm', setup_dir)
        write_status(setup_dir, True, 'success')
    except Exception as e:
        write_status(setup_dir, False, render_exception(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--setup-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.setup_dir)
