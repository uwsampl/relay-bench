import os
import numpy as np
import mxnet as mx
from mxnet import gluon

from common import idemp_mkdir

INPUT_SIZE = 128
HIDDEN_SIZE = 128
SEQ_LEN = 5
BATCH = 1

class RNNModel(gluon.HybridBlock):
    def __init__(self, cell_type, input_size, hidden_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.input_size = input_size
            self.hidden_size = hidden_size
            if cell_type == 'rnn':
                self.cell = gluon.rnn.RNNCell(hidden_size, input_size=input_size)
            elif cell_type == 'gru':
                self.cell = gluon.rnn.GRUCell(hidden_size, input_size=input_size)
            elif cell_type == 'lstm':
                self.cell = gluon.rnn.LSTMCell(hidden_size, input_size=input_size)
            else:
                raise RuntimeError("Unsupported RNN cell type: %s" % cell_type)

    def hybrid_forward(self, F, xs, states):
        body = lambda data, states: self.cell(data, states)
        outs, states = F.contrib.foreach(body, xs, states)
        return outs, states


def model_filename(cell_type):
    return '{}_i{}_h{}'.format(cell_type, INPUT_SIZE, HIDDEN_SIZE)


def export_mxnet_model(cell_type, setup_dir):
    # batch and seq_len are placeholder, and don't affect the exported model
    ctx = mx.context.cpu()
    dtype = 'float32'
    model = RNNModel(cell_type, INPUT_SIZE, HIDDEN_SIZE)
    if cell_type == 'rnn' or cell_type == 'gru':
        states = [mx.nd.zeros((BATCH, HIDDEN_SIZE), dtype=dtype, ctx=ctx)]
    elif cell_type == 'lstm':
        states = [mx.nd.zeros((BATCH, HIDDEN_SIZE), dtype=dtype, ctx=ctx),
                  mx.nd.zeros((BATCH, HIDDEN_SIZE), dtype=dtype, ctx=ctx)]
    xs = mx.nd.random.uniform(shape=(SEQ_LEN, BATCH, INPUT_SIZE),
                              dtype=dtype, ctx=ctx)

    model.collect_params().initialize(ctx=ctx)
    model.hybridize()
    model(xs, states)
    idemp_mkdir(os.path.join(setup_dir, 'mxnet'))
    fname = os.path.join(setup_dir, 'mxnet', model_filename(cell_type))
    model.export(fname, epoch=1)
    print('Export MXNet model to %s' % fname)
    return fname


def import_gluon_rnn(name):
    if name not in ['rnn', 'gru', 'lstm']:
        raise Exception('Unsupported network: %s' % name)
    data_shape = (SEQ_LEN, BATCH, INPUT_SIZE)
    state_shape = (BATCH, HIDDEN_SIZE)
    num_states = 2 if name == 'lstm' else 1

    data_names = ['data%s' % i for i in range(num_states + 1)]
    shapes = {'data' : data_shape}
    for i in range(num_states):
        shapes['state%s' % i] = state_shape

    fname = model_filename(name)
    net = gluon.nn.SymbolBlock.imports(
        "setup/mxnet/%s-symbol.json" % (fname),
        data_names,
        "setup/mxnet/%s-0001.params" % (fname))
    return net, num_states, shapes
