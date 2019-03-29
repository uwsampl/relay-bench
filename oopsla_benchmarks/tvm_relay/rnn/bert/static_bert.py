from tvm import relay
import mxnet as mx
from mxnet import gluon

# In order to run, execute export_bert.py, and then
# place the model and params at the below location.
net = gluon.nn.SymbolBlock.imports(
    "gluon_export/bert_static.json",
    ["data0", "data1", "data2"],
    "gluon_export/bert_static.params")

data0 = mx.sym.Variable('data0')
data1 = mx.sym.Variable('data1')
data2 = mx.sym.Variable('data2')
expr, params = relay.frontend.from_mxnet(
    net,
    shape={'data0': (24, 384), 'data1': (24, 384), 'data2': (24,) },
    input_symbols=[data0, data1, data2])

import pdb; pdb.set_trace()
