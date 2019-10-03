from common import invoke_main, render_exception, write_status
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
    invoke_main(main, 'config_dir', 'setup_dir')
