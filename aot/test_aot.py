from tvm import relay
from tvm.relay import var, Function, op
import numpy as np
import tvm
import aot

def convert(val, dtype=None):
    return tvm.nd.array(np.array(val).astype(dtype))

def test_add():
    x = var('x', shape=())
    y = var('y', shape=())
    z = x + y
    func = Function([x, y], z)
    cfunc = aot.compile(func)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    c = tvm.nd.array(np.array(2.0, dtype='float32'))
    output = cfunc(a, b)
    np.testing.assert_allclose(output.asnumpy(), c.asnumpy())

def test_mult_op():
    x = var('x', shape=())
    y = var('y', shape=())
    z = x + y
    zz = op.exp(z)
    func = Function([x, y], zz)
    cfunc = aot.compile(func)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    output = cfunc(a, b)
    np.testing.assert_allclose(output.asnumpy(), np.exp(a.asnumpy() + b.asnumpy()))

if __name__ == "__main__":
    # test_add()
    # import pdb; pdb.set_trace()
    test_mult_op()

