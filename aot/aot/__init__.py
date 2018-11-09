import ctypes
import numpy as np
import os
import subprocess
import tvm
from tvm import relay, get_global_func, target, register_func
from tvm.relay.expr import ExprFunctor, Expr, Let
from tvm.relay.backend import compile_engine
from .little_cpp import PackedCall, CPPFunction, Invoke
from . import to_source

TVM_PATH = os.environ['TVM_PATH']


def compile_cpp(source, lib_name, lib_path=None):
    if lib_path is None:
        lib_path = os.curdir

    source_path = os.path.join(lib_path, 'source.cc')
    with open(source_path, 'w') as source_file:
        source_file.write(source)

    source_file.close()

    system = os.uname()[0]
    if system == 'Darwin':
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
    else:
        command = [
            "clang",
            "-std=c++14",
            "-shared",
            "-undefined",
            "dynamic_lookup",
            "-fPIC",
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
    cleanup = [
        "rm",
        source_path
    ]
    assert subprocess.run(cleanup).returncode == 0


def load_lib(name):
    return ctypes.CDLL(name, ctypes.RTLD_GLOBAL)

# example_src = build_source()
# compile_cpp(example_src, "libexample.so")
# load_lib("libexample.so")
# get_global_func("relay.aot.example")("Hello Relay")


def is_primitive(func: relay.Function):
    return isinstance(func, relay.Function) and func.attrs and func.attrs.Primitive.value == 1

class AoTCompiler(ExprFunctor):
    def __init__(self) -> None:
        super().__init__()
        self.engine = compile_engine.get()
        self.bindings = [[]]

    def add_binding(self, var, value):
        self.bindings[-1].append((var, value))

    def optimize(self, expr: Expr) -> Expr:
        infer_e = relay.ir_pass.infer_type(expr)
        fused_e = relay.ir_pass.fuse_ops(infer_e)
        fused_e = relay.ir_pass.infer_type(fused_e)
        anf_fused = relay.ir_pass.to_anf(fused_e)
        anf_fused = relay.ir_pass.infer_type(anf_fused)
        return anf_fused

    def mk_primitive_op(self, func: Expr, args, output_type) -> Expr:
        cc_key = compile_engine.CCacheKey(func, target.create('llvm'))
        jit_func = self.engine.jit(cc_key)
        hash = relay.ir_pass.structural_hash(func)
        name = f"op{hash}"
        if not get_global_func(name, allow_missing=True):
            register_func(name, jit_func)
        return PackedCall(name, len(func.params) + 1, args, output_type)

    def visit_call(self, call: Expr) -> Expr:
        if is_primitive(call.op):
            return self.mk_primitive_op(call.op, call.args, call.checked_type)
        else:
            args = [self.visit(arg) for arg in call.args]
            fn = self.visit(call.op)
            return Invoke(fn, args)

    def visit_let(self, let: Expr) -> Expr:
        self.bindings.append([])

        while isinstance(let, Let):
            cpp_value = self.visit(let.value)
            self.add_binding(let.var, cpp_value)
            let = let.body

        bindings = self.bindings.pop()
        body = self.visit(let)

        return little_cpp.Decl(bindings, body)

    def visit_var(self, var):
        return var

    def visit_function(self, func):
        if is_primitive(func):
            body = self.mk_primitive_op(func, func.params, func.ret_type)
            return CPPFunction(func.params, body, func.checked_type)
        else:
            return CPPFunction(func.params, self.visit(func.body), func.checked_type)

_LIB_COUNTER = 1
_LIB = []

def compile(func, name='default'):
    global _LIB, _LIB_COUNTER
    packed_name = f'relay.aot.{name}.{_LIB_COUNTER}'
    compiler = AoTCompiler()
    func = compiler.optimize(func)
    func = compiler.visit(func)
    source_code = to_source.to_source(packed_name, func)
    lib_name = f"librelay_aot_{_LIB_COUNTER}.so"
    _LIB_COUNTER += 1
    compile_cpp(source_code, lib_name)
    _LIB.append(load_lib(lib_name))
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    fn = get_global_func(packed_name)
    return fn

