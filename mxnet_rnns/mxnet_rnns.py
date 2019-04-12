import csv
import os
import time
import mxnet as mx
from mxnet import gluon

import tvm
import numpy as np
from tvm import relay
from tvm.relay.vm import _eval_vm, eta_expand
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.prelude import Prelude
import aot

def vm_eval(f, *args, ctx=tvm.cpu()):
    if isinstance(f, relay.Expr):
        ex = relay.create_executor('vm', mod=relay.Module(), ctx=ctx)
        if len(args) == 0:
            return ex.evaluate(f)
        else:
            return ex.evaluate(f)(*args)
    else:
        assert isinstance(f, relay.Module), "expected expression or module"
        mod = f
        ex = relay.create_executor('vm', mod=mod, ctx=ctx)
        if len(args) == 0:
            return ex.evaluate(mod[mod.entry_func])
        else:
            return ex.evaluate(mod[mod.entry_func])(*args)

def import_mxnet_model(fname, num_states):
    ctx = mx.context.cpu()
    data_names = ['data0']
    for i in range(num_states):
        data_names.append('data%s' % (i+1))

    model_data_dir = os.path.dirname(os.path.realpath(__file__))
    net = gluon.nn.SymbolBlock.imports("%s/model_zoo_data/%s-symbol.json.data" % (model_data_dir, fname), data_names,
                                       "%s/model_zoo_data/%s-0001.params.data" % (model_data_dir, fname), ctx=ctx)
    net.hybridize()
    return net

def import_net(cell_type):
    num_states = 2 if cell_type == 'lstm' else 1
    return import_mxnet_model("%s_i128_h128" % cell_type, num_states)

def create_relay_aot_func(mx_net, num_states=1):
    input_size = 128
    hidden_size = 128
    batch, seq_len = 1, 5
    data_shape= (seq_len, batch, input_size)
    state_shape = (batch, hidden_size)

    shapes = {'data': data_shape}
    mx_input_syms = []
    mx_input_syms.append(mx.sym.Variable("data"))
    for i in range(num_states):
        shapes['state%s' % i] = state_shape
        mx_input_syms.append(mx.sym.Variable("state%s" % i))

    mod = relay.module.Module()
    relay_net, params = relay.frontend.from_mxnet(mx_net, shape=shapes, input_symbols=mx_input_syms, module=mod)
    params = params.items()

    loop = None
    for v, func in mod.functions.items():
        if v.name_hint == 'loop':
            loop = v

    inputs = [relay.var('data')]
    for i in range(num_states):
        inputs.append(relay.var('state%s' % i))
    for name, _ in params:
        inputs.append(relay.var(name))
    mod[mod.entry_func] = relay.Function(inputs, relay.Call(relay_net, inputs))

    use_gpu = False
    context = tvm.gpu(0) if use_gpu else tvm.cpu(0)
    target = tvm.target.cuda() if use_gpu else tvm.target.create('llvm')
    relay_aot_func = aot.compile(mod, mod.entry_func, ctx=context, tgt=target, use_gpu=use_gpu)
    return relay_aot_func


def test_rnn(relay_aot_func, mx_net, num_states=1):
    input_size = 128
    hidden_size = 128
    batch, seq_len = 1, 5
    data_shape= (seq_len, batch, input_size)
    state_shape = (batch, hidden_size)

    shapes = {'data': data_shape}
    mx_input_syms = []
    mx_input_syms.append(mx.sym.Variable("data"))
    for i in range(num_states):
        shapes['state%s' % i] = state_shape
        mx_input_syms.append(mx.sym.Variable("state%s" % i))

    mod = relay.module.Module()
    relay_net, params = relay.frontend.from_mxnet(mx_net, shape=shapes, input_symbols=mx_input_syms, module=mod)
    params = params.items()

    loop = None
    for v, func in mod.functions.items():
        if v.name_hint == 'loop':
            loop = v

    inputs = [relay.var('data')]
    for i in range(num_states):
        inputs.append(relay.var('state%s' % i))
    for name, _ in params:
        inputs.append(relay.var(name))
    mod[mod.entry_func] = relay.Function(inputs, relay.Call(relay_net, inputs))

    #use_gpu = False
    #context = tvm.gpu(0) if use_gpu else tvm.cpu(0)
    #target = tvm.target.cuda() if use_gpu else tvm.target.create('llvm')
    #relay_aot_func = aot.compile(mod, mod.entry_func, ctx=context, tgt=target, use_gpu=use_gpu)

    data_v = np.random.rand(seq_len, batch, 128).astype('float32')
    states_v = [np.random.rand(*state_shape).astype('float32') for _ in range(num_states)]
    params_v = [e[1].asnumpy() for e in params]

    start = time.time()
    result = relay_aot_func(data_v, *states_v, *params_v)
    end = time.time()
    relay_time = end - start

    mx_inputs = [mx.nd.array(x) for x in [data_v, *states_v]]
    start = time.time()
    mx_outputs = mx_net(*mx_inputs)[0].asnumpy()
    end = time.time()
    mx_net_time = end - start

    return relay_time, mx_net_time


if __name__ == "__main__":
    NUM_THROWAWAY = 8
    NUM_ITERS = 10000
    EXPER_TIMEOUT = 10800 / 3
    #EXPER_TIMEOUT = 10

    rnn_model = import_net('rnn')
    gru_model = import_net('gru')
    lstm_model = import_net('lstm')

    relay_aot_rnn = create_relay_aot_func(rnn_model)
    relay_aot_gru = create_relay_aot_func(gru_model)
    relay_aot_lstm = create_relay_aot_func(lstm_model, 2)

    relay_rnn_times = []
    relay_gru_times = []
    relay_lstm_times = []
    mx_net_rnn_times = []
    mx_net_gru_times = []
    mx_net_lstm_times = []
    print(f'running with {NUM_ITERS} iterations')
    global_start = time.time()
    for i in range(NUM_ITERS):
        relay_time, mx_net_time = test_rnn(relay_aot_rnn, rnn_model)
        if i < NUM_THROWAWAY:
            relay_rnn_times.append(relay_time)
            mx_net_rnn_times.append(mx_net_time)
            print(f'relay: {relay_time} seconds, mx_net: {mx_net_time} seconds')

        relay_time, mx_net_time = test_rnn(relay_aot_gru, gru_model)
        if i < NUM_THROWAWAY:
            relay_rnn_times.append(relay_time)
            mx_net_rnn_times.append(mx_net_time)
            print(f'relay: {relay_time} seconds, mx_net: {mx_net_time} seconds')

        relay_time, mx_net_time = test_rnn(relay_aot_lstm, lstm_model, 2)
        if i < NUM_THROWAWAY:
            relay_rnn_times.append(relay_time)
            mx_net_rnn_times.append(mx_net_time)
            print(f'relay: {relay_time} seconds, mx_net: {mx_net_time} seconds')

        running_time = time.time() - global_start
        if running_time > EXPER_TIMEOUT:
            print(f'Experiment timed out after {i}/{NUM_ITERS} trials and {running_time} seconds')
            break

    # print('[RNN]')
    # rnn_times = collect_timing(lambda _: , num_iters=NUM_ITERS)

    # print('[GRU]')
    # gru_times = collect_timing(lambda _: test_rnn(gru_model), num_iters=NUM_ITERS)

    # print('[LSTM]')
    # lstm_times = collect_timing(lambda _: test_rnn(lstm_model, 2), num_iters=NUM_ITERS)

    with open('mxnet_rnn_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Trial Number', 'RNN (Relay)', 'GRU (Relay)', 'LSTM (Relay)', 'RNN (MxNet)', 'GRU (MxNet)', 'LSTM (MxNet)'])
        for i, curr_row in enumerate(zip(relay_rnn_times, relay_gru_times, relay_lstm_times, mx_net_rnn_times, mx_net_gru_times, mx_net_lstm_times)):
            writer.writerow([i] + list(curr_row))


