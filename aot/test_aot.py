from tvm.relay import var, Function
import aot

def test_simple():
    x = var('x', shape=())
    y = var('y', shape=())
    z = x + y
    func = Function([x, y], z)
    aot.compile(func)

if __name__ == "__main__":
    test_simple()

