from tvm import relay
import mxnet as mx
from mxnet import gluon
import os
import subprocess
from tvm.relay.ir_pass import infer_type

own_dir = os.path.dirname(__file__)

if not os.path.exists(os.path.join(own_dir, "gluon_export/bert_static-symbol.json")):
    subprocess.run(["python3",  "export_bert.py", "--output_dir", "../../gluon_export"],
        cwd=os.path.join(own_dir, 'gluon_bert/staticbert/'))

# In order to run, execute export_bert.py, and then
# place the model and params at the below location.
net = gluon.nn.SymbolBlock.imports(
    os.path.join(own_dir, "gluon_export/bert_static-symbol.json"),
    ["data0", "data1", "data2"],
    os.path.join(own_dir, "gluon_export/bert_static-0003.params"))

data0 = mx.sym.Variable('data0', shape=(24, 384))
data1 = mx.sym.Variable('data1', shape=(24, 384))
data2 = mx.sym.Variable('data2', shape=(24,))
# data0 = mx.ndarray.zeros((24, 384))
# data1 = mx.ndarray.zeros((24, 384))
# data2 = mx.ndarray.zeros((24,))
# net = net(data0, data1, data2)

expr, params = relay.frontend.from_mxnet(
    net,
    shape={'data0': (24, 384), 'data1': (24, 384), 'data2': (24,) },
    input_symbols=[data0, data1, data2])

expr = relay.ir_pass.infer_type(expr)

model = (expr, params)
