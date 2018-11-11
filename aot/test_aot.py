from tvm import relay
from tvm.relay import var, Function, op, Module
from tvm.relay.prelude import Prelude
import numpy as np
import tvm
import aot

mod = Module()
p = Prelude(mod)

#print(aot.do_type(mod, p.nat))
#raise

def test_add():
    x = var('x', shape=())
    y = var('y', shape=())
    z = x + y
    func = Function([x, y], z)
    cfunc = aot.compile(mod, func)
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
    cfunc = aot.compile(mod, func)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    output = cfunc(a, b)
    np.testing.assert_allclose(output.asnumpy(), np.exp(a.asnumpy() + b.asnumpy()))

if __name__ == "__main__":
    # test_add()
    test_mult_op()
