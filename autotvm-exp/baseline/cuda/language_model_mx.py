"""
Benchmark single rnn cell and language model (embedding + rnn + decoder)
"""

from collections import namedtuple
import argparse
import logging
import time

import numpy as np

import mxnet as mx
import mxnet.rnn.rnn_cell as rnn_cell

import tvm
from util import log_value, array2str_round

wkls = [
#    ('RNN.B4.L2.S1.H650.V0',      'rnn',  4, 2, 1, 650, 0),
#    ('RNN.B4.L2.S1.H650.V10000',  'rnn',  4, 2, 1, 650, 10000),

    ('LSTM.B4.L2.S1.H650.V0',     'lstm', 4, 2, 1, 650, 0),
    ('LSTM.B4.L2.S1.H650.V10000', 'lstm', 4, 2, 1, 650, 10000),
]

def get_cell(cell_type, batch_size, hidden_size, fused, prefix):
    if fused:
        cell_type += '-fused'

    if cell_type == 'rnn':
        cell = rnn_cell.RNNCell(num_hidden=hidden_size, prefix=prefix)
    elif cell_type == 'lstm':
        cell = rnn_cell.LSTMCell(num_hidden=hidden_size, prefix=prefix)
    elif cell_type == 'rnn-fused':
        cell = rnn_cell.FusedRNNCell(num_hidden=hidden_size, prefix=prefix,
                                     mode='rnn_tanh', get_next_state=True)
    elif cell_type == 'lstm-fused':
        cell = rnn_cell.FusedRNNCell(num_hidden=hidden_size, prefix=prefix,
                                     mode='lstm', get_next_state=True)
    else:
        raise RuntimeError("Invalid cell type " + cell_type)

    return cell

def measure_cell(cell_wkl, fused, n_times):
    task_name, cell_type, batch_size, num_layer, seq_len, hidden_size, voc_size = cell_wkl

    ctx = mx.gpu()

    # encoder
    data = mx.sym.Variable('data')

    if voc_size:
        weight = mx.sym.var("encoder_weight")
        embed = mx.sym.Embedding(data=data, weight=weight, input_dim=voc_size,
                                 output_dim=hidden_size, name='embed')
    else:
        embed = data

    states = []
    state_names = []
    outputs = embed
    for i in range(num_layer):
        prefix = 'cell_l%d_' % i
        cell = get_cell(cell_type, batch_size, hidden_size, fused, prefix)
        outputs, states = cell.unroll(seq_len, inputs=outputs,
                                      merge_outputs=True, layout='TNC')

    if voc_size:
        outputs = mx.sym.Reshape(outputs, shape=(-1, hidden_size))
        outputs = mx.sym.FullyConnected(data=outputs, num_hidden=voc_size)
        outputs = mx.sym.Reshape(outputs, shape=(-1, seq_len, voc_size))

    if isinstance(outputs, (list, tuple)):
        output = mx.sym.Group(outputs)
    else:
        output = outputs

    mod = mx.mod.Module(symbol=output,
                        context=ctx,
                        data_names=['data'],
                        label_names=None)

    if voc_size:
        data = mx.io.DataBatch(data=[mx.nd.array(np.random.randint(0, voc_size,
                                                 (seq_len, batch_size)),
                                                 ctx=ctx)],
                               provide_data=[mx.io.DataDesc('data', (seq_len, batch_size),
                                                            dtype=np.int32)])
    else:
        data = mx.io.DataBatch(data=[mx.nd.array(np.random.randn(seq_len, batch_size, hidden_size).astype(np.float32),
                                                 ctx=ctx)],
                               provide_data=[mx.io.DataDesc('data', (seq_len, batch_size, hidden_size),
                                                            dtype=np.float32)])


    mod.bind(data_shapes=data.provide_data)
    mod.init_params()

    for _ in range(5):
        mod.forward(data, is_train=False)
    mx.nd.waitall()

    tic = time.time()
    for i in range(n_times):
        mod.forward(data, is_train=False)
        for x in mod.get_outputs():
            x.asnumpy()
    costs = time.time() - tic

    return costs / n_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-ave-curve', type=int, default=5)
    parser.add_argument("--n-times", type=int, default=1000,
                                     help="number of runs to take average for time cost")
    parser.add_argument("--dtype", type=str, default='float32')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    logging.info(args)


    for wkl in wkls:
        for fused in [True, False]:
            task_name = wkl[0]
            while True:
                costs = []
                for i in range(args.n_ave_curve):
                    tvm_res = measure_cell(wkl, fused, args.n_times)
                    costs.append(tvm_res)

                if np.std(costs) / np.mean(costs) < 0.03:
                    break
                print(array2str_round(costs), "retry due to high variance of measure results...")

            print(wkl, array2str_round(costs))
            device_name = tvm.gpu(0).device_name
            method = 'mxnet-fused' if fused else 'mxnet'
            log_value('cuda', device_name, task_name, method,
                      array2str_round(costs))

