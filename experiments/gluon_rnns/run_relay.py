import sys

from validate_config import validate
from common import invoke_main, write_status
from trial_util import run_trials, configure_seed

import numpy as np

import mxnet as mx
from mxnet_util import import_gluon_rnn

import tvm
from tvm import relay

import aot

def rnn_setup(device, method, network):
    mx_net, num_states, shapes = import_gluon_rnn(network)
    use_gpu = (device == 'gpu')
    use_aot = (method == 'aot')

    input_symbols = [mx.sym.Variable('data')] + [mx.sym.Variable('state%s' % i)
                                                 for i in range(num_states)]

    relay_net, params = relay.frontend.from_mxnet(mx_net, shape=shapes)
    params = params.items()

    inputs = [
        relay.var('data')] + [
            relay.var('state%s' % i) for i in range(num_states)] + [
            relay.var(pair[0]) for pair in params]
    relay_func = relay_net['main']
    relay_net['main'] = relay.Function(inputs, relay.Call(relay_func, inputs))

    context = tvm.gpu(0) if use_gpu else tvm.cpu(0)
    target = tvm.target.cuda() if use_gpu else tvm.target.create('llvm')

    data_v = np.random.rand(*shapes['data']).astype('float32')
    states_v = [np.random.rand(*shapes['state%s' % i]).astype('float32')
                for i in range(num_states)]
    params_v = [pair[1].asnumpy() for pair in params]

    if use_aot:
        func = aot.compile(relay_net['main'], relay_net, ctx=context, tgt=target)
    else:
        executor = relay.create_executor(mod=relay_net, ctx=context, target=target)
        func = executor.evaluate(relay_net['main'])
    thunk = lambda: func(data_v, *states_v, *params_v)
    return [thunk]


def rnn_trial(thunk):
    return thunk()


def rnn_teardown(thunk):
    pass


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'relay' not in config['frameworks']:
        write_status(output_dir, True, 'Relay not run')
        sys.exit(0)

    configure_seed(config)

    success, msg = run_trials(
        'relay', 'gluon_rnns',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        rnn_trial, rnn_setup, rnn_teardown,
        ['device', 'method', 'network'],
        [config['devices'], config['relay_methods'], config['networks']],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'output_dir')
