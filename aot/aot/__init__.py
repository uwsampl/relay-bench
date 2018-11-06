import ctypes
import numpy as np
import os
import subprocess
import tvm
from tvm import relay, get_global_func, target, register_func
from tvm.relay.expr import ExprFunctor
from tvm.relay.backend import compile_engine
from .little_cpp import PackedCall, CPPFunction
from . import to_source

TVM_PATH = os.environ['TVM_PATH']



def compile_cpp(source, lib_name, lib_path=None):
    if lib_path is None:
        lib_path = os.curdir

    source_path = os.path.join(lib_path, 'source.cc')
    with open(source_path, 'w') as source_file:
        source_file.write(source)

    command = [
        "clang",
        "-std=c++14",
        "-shared",
        "-undefined",
        "dynamic_lookup",
        "-o",
        lib_name,
        source_path,
		f"-I{TVM_PATH}/3rdparty/dmlc-core/include",
		f"-I{TVM_PATH}/3rdparty/dlpack/include",
		f"-I{TVM_PATH}/3rdparty/HalideIR/src",
		f"-I{TVM_PATH}/include",
		f"-L{TVM_PATH}/build",
        "-ltvm"
    ]

    proc = subprocess.run(command)
    assert proc.returncode == 0


def load_lib(name):
    return ctypes.CDLL(name, ctypes.RTLD_GLOBAL)

# example_src = build_source()
# compile_cpp(example_src, "libexample.so")
# load_lib("libexample.so")
# get_global_func("relay.aot.example")("Hello Relay")


def is_primitive(func: relay.Function):
    return isinstance(func, relay.Function) and func.attrs and func.attrs.Primitive.value == 1

class AoTCompiler(ExprFunctor):
    def __init__(self):
        super().__init__()
        self.engine = compile_engine.get()

    def optimize(self, expr):
        infer_e = relay.ir_pass.infer_type(expr)
        fused_e = relay.ir_pass.fuse_ops(infer_e)
        fused_e = relay.ir_pass.infer_type(fused_e)
        return fused_e

    def mk_primitive_op(self, func, args, output_type):
        cc_key = compile_engine.CCacheKey(func, target.create('llvm'))
        jit_func = self.engine.jit(cc_key)
        hash = relay.ir_pass.structural_hash(func)
        name = f"op{hash}"
        register_func(name, jit_func)
        return PackedCall(name, len(func.params) + 1, args, output_type)

    def visit_call(self, call):
        if is_primitive(call.op):
            return self.mk_primitive_op(call.op, call.args, call.checked_type)
        else:
            raise Exception("...")

    def visit_var(self, var):
        return var

    def visit_function(self, func):
        if is_primitive(func):
            return self.mk_primitive_op(func, call.args, func.ret_type)
        else:
            return CPPFunction(func.params, self.visit(func.body), func.checked_type)

_LIB = None

def compile(func, name='default'):
    global _LIB
    packed_name = f'relay.aot.{name}'
    compiler = AoTCompiler()
    func = compiler.optimize(func)
    func = compiler.visit(func)
    source_code = to_source.to_source(packed_name, func)
    lib_name = "libtest.so"
    compile_cpp(source_code, "libtest.so")
    _LIB = load_lib("libtest.so")
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    fn = get_global_func(packed_name)
    return fn
