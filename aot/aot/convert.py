import numpy as np
import tvm
from tvm import relay

# convert(convert(a)) = convert(a)
def convert(a):
    if isinstance(a, int):
        return convert(np.array(a, dtype='int32'))
    elif isinstance(a, np.ndarray):
        return convert(tvm.nd.array(a))
    elif isinstance(a, tvm.ndarray.NDArray):
        return relay.backend.interpreter.TensorValue(a)
    elif isinstance(a, relay.Call):
        assert isinstance(a.op, relay.Constructor)
        return relay.backend.interpreter.ConValue(a.op, [convert(arg) for arg in a.args], [])
    elif isinstance(a, relay.backend.interpreter.TensorValue):
        return a
    elif isinstance(a, relay.backend.interpreter.ConValue):
        return a
    else:
        raise Exception(a, type(a))
