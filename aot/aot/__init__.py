import ctypes
import numpy as np
import os
import subprocess
import tvm
from tvm import relay, get_global_func, target, register_func
from tvm.relay.expr import ExprFunctor
from tvm.relay.backend import compile_engine
import little_cpp

TVM_PATH = os.environ['TVM_PATH']

def mk_file(body):
    return f"""
    #include <tvm/tvm.h>
    #include <tvm/api_registry.h>

    using namespace tvm;

    {body}
    """

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
    return func.attrs and func.attrs.Primitive

class AoTCompiler(ExprFunctor):
    def __init__():
        self.engine = compile_engine.get()

    def optimize(self, expr):
        infer_e = relay.ir_pass.infer_type(expr)
        fused_e = relay.ir_pass.fuse_ops(infer_e)
        return fused_e

    def mk_primitive_op(self, func):
        cc_key = compile_engine.CCacheKey(func, target.create('llvm'))
        jit_func = self.engine.jit(cc_key)
        return PackedCall(jit_func, len(func.params) + 1)

    def visit_call(self, call):
        if is_primitive(call.op):
            return self.mk_primitive_op(call.op)
        else:
            raise Exception("...")

    def visit_var(self, var):
        return var

    def visit_function(self, func):
        import pdb; pdb.set_trace()
        if is_primitive(func):
            return self.mk_primitive_op(func)
        else:
            self.visit(func.body)

def mk_func_api_wrapper(name, jit_func, arity):
    args = ""
    for i in range(arity):
        args += f"args[{i}]"
        if i != arity - 1:
            args += ", "

    return f"""
    TVM_REGISTER_API("{name}")
    .set_body([](TVMArgs args, TVMRetValue* ret) {{
        PackedFunc *pf = reinterpret_cast<PackedFunc*>({jit_func.handle.value});
        CHECK(pf);
        (*pf)({args});
    }});
    """

def compile(func):
    compiler = AoTCompiler()
    func = compiler.optimize(func)
    func = compiler.visit(func)
    engine = compile_engine.get()
    cc_key = compile_engine.CCacheKey(func, target.create('llvm'))
    jit_func = engine.jit(cc_key)
    wrapper = mk_func_api_wrapper('aaa', jit_func, 3)
    source = mk_file(wrapper)
    lib_name = "libtest.so"
    compile_cpp(source, "libtest.so")
    load_lib("libtest.so")
    a = tvm.nd.array(np.array(1).astype('float32'))
    b = tvm.nd.array(np.array(2).astype('float32'))
    c = tvm.nd.array(np.array(0).astype('float32'))
    fn = get_global_func('aaa')
    import pdb; pdb.set_trace()

