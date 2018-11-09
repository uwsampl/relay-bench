from . import little_cpp
from tvm import relay

#  PackedFunc *pf = reinterpret_cast<PackedFunc*>({jit_func.handle.value});
#         CHECK(pf);
#         (*pf)({args});

class ToSource:
    def __init__(self):
        self.name_counter = 0
        self.source_content = ""
        self.name_map = {}
        self.cont = []
        self.local = True

    def fresh_global_name(self):
        name = f"global{self.name_counter}"
        self.name_counter += 1
        return name

    def fresh_local_name(self, var):
        name = f"local_{var.name_hint}_{self.name_counter}"
        self.name_counter += 1
        return name

    def do_cont(self, *args):
        cont = self.cont.pop()
        return cont(*args)

    def visit(self, node, local=True):
        old_local = self.local
        self.local = local

        if isinstance(node, little_cpp.PackedCall):
            res = self.visit_packed_call(node)
        elif isinstance(node, little_cpp.CPPFunction):
            res = self.visit_cpp_function(node)
        elif isinstance(node, little_cpp.Decl):
            res = self.visit_decl(node)
        elif isinstance(node, little_cpp.Invoke):
            res = self.visit_invoke(node)
        elif isinstance(node, relay.Var):
            res = self.name_map[node]
        else:
            raise Exception("...")

        self.local = old_local

        return res

    def visit_invoke(self, invoke):
        args_str = ""
        for i, arg in enumerate(invoke.args):
            assert isinstance(arg, relay.Var)
            args_str += self.visit(arg)
            if i != len(invoke.args) - 1:
                args_str += ", "

        func = self.visit(invoke.call)
        return f"{func}({args_str})"

    def visit_decl(self, decl):
        source = ""
        for var, value in decl.bindings:
            local_name = self.fresh_local_name(var)
            self.name_map[var] = local_name
            value_str = self.visit(value)
            source += f"auto {local_name} = {value_str};\n"
        source += self.do_cont(self.visit(decl.body))
        return source

    def visit_packed_call(self, call):
        args = ""
        end = len(call.args) - 1
        for i, arg in enumerate(call.args):
            varg = self.visit(arg)
            args += self.name_map[arg]
            if i != end:
                args += ", "

        return f"""
            const PackedFunc *pf = runtime::Registry::Get("{call.name}");
            CHECK(pf);
            NDArray out = NDArray::Empty({{}}, dtype_f32, context);
            (*pf)({args}, out);
            {self.do_cont("out")};
        """

    def visit_cpp_function(self, func):
        name = func.name = self.fresh_global_name()
        param_str = ""

        end = len(func.params) - 1
        for i, param in enumerate(func.params):
            pname = f"param{i}"
            self.name_map[param] = pname
            param_str += f"const NDArray& {pname}"
            if i != end:
                param_str += ", "

        self.cont.append(lambda end: f"return {end};")

        body = self.visit(func.body)

        if self.local:
            func = f"""[&]({param_str}) {{
                {body}
            }}
            """
        else:
            func = f"""
            NDArray {name}({param_str}) {{
                {body}
            }}
            """
        return func

    def mk_register_api(self, name: str, func: little_cpp.CPPFunction) -> str:
        assert isinstance(func, little_cpp.CPPFunction)
        source = ""
        source += self.visit(func, False)

        args = ""
        end = len(func.params) - 1
        for i in range(len(func.params)):
            args += f"args[{i}]"
            if i != end:
                args += ", "

        source += f"""
        TVM_REGISTER_API("{name}")
        .set_body([](TVMArgs args, TVMRetValue* ret) {{
            *ret = {func.name}({args});
        }});
        """
        return source

def mk_file(body):
    return f"""
    #include <tvm/tvm.h>
    #include <tvm/api_registry.h>

    using namespace tvm;
    using namespace runtime;

    static DLDataType dtype_f32 = DLDataType {{ .code = 2, .bits = 32, .lanes = 1 }};
    static DLContext context = DLContext {{ .device_type = DLDeviceType::kDLCPU, . device_id = 0 }};
    {body}
    """

def to_source(name, program) -> str:
    assert isinstance(program, little_cpp.CPPFunction)
    convert = ToSource()
    return mk_file(convert.mk_register_api(name, program))
