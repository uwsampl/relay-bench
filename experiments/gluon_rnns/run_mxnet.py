import argparse
import sys

from validate_config import validate
from common import write_status
from trial_util import run_trials, configure_seed

import numpy as np
from collections import namedtuple
import mxnet as mx

from mxnet_util import import_gluon_rnn

def rnn_setup(dev, network):
    use_gpu = (dev == 'gpu')
    context = mx.gpu(0) if use_gpu else mx.cpu(0)

    net, num_states, shapes = import_gluon_rnn(network)
    net.initialize(ctx=context)
    net.hybridize()

    shape_list = [shapes['data']] + [shapes['state%s' % i] for i in range(num_states)]
    mx_inputs = [mx.nd.array(np.random.rand(*shape).astype('float32'), ctx=context)
                 for shape in shape_list]
    thunk = lambda: net(*mx_inputs)[0].asnumpy()
    return [thunk]


def rnn_trial(thunk):
    thunk()


def rnn_teardown(thunk):
    pass


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'mxnet' not in config['frameworks']:
        write_status(output_dir, True, 'MxNet not run')
        sys.exit(0)

    configure_seed(config)

    success, msg = run_trials(
        'mxnet', 'gluon_rnns',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        rnn_trial, rnn_setup, rnn_teardown,
        ['device', 'network'],
        [config['devices'], config['networks']],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir)
