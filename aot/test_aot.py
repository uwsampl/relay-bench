from tvm import relay
from tvm.relay import var, Function, op, Module, GlobalVar
from tvm.relay.prelude import Prelude
import numpy as np
import tvm
import aot

#print(aot.do_type(mod, p.nat))
#raise

def test_identity():
    mod = Module()
    x = var('x', shape=())
    func = Function([x], x)
    cfunc = aot.compile(mod, func)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), a.asnumpy())

def test_add():
    mod = Module()
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
    mod = Module()
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

def test_double():
    mod = Module()
    x = var('x', shape=())
    double = GlobalVar('double')
    mod[double] = Function([x], x + x)
    x = var('x', shape=())
    cfunc = aot.compile(mod, Function([x], double(double(x))))
    a = tvm.nd.array(np.array(1.5, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(6.0, dtype='float32'))

if __name__ == "__main__":
    #test_identity()
    #test_add()
    #test_mult_op()
    test_double()
