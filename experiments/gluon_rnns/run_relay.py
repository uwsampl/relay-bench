from validate_config import validate
from exp_templates import (common_trial_params, common_early_exit,
                           run_template)

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


if __name__ == '__main__':
    run_template(validate_config=validate,
                 check_early_exit=common_early_exit({'frameworks': 'relay'}),
                 gen_trial_params=common_trial_params(
                     'relay', 'gluon_rnns',
                     rnn_trial, rnn_setup, rnn_teardown,
                     ['device', 'method', 'network'],
                     ['devices', 'relay_methods', 'networks']))
