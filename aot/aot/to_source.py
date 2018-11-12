from . import little_cpp
from tvm import relay

class ToSource:
    def __init__(self, gv_map):
        self.gv_map = gv_map
        self.name_counter = 0
        self.source_content = ""
        self.name_map = {}
        self.local = True
        self.declare = ""
        self.declare_map = {}

    def fresh_global_name(self):
        name = f"global{self.name_counter}"
        self.name_counter += 1
        return name

    def fresh_local_name(self, var=None):
        if var is not None:
            name = f"local_{var.name_hint}_{self.name_counter}"
        else:
            name = f"local_{self.name_counter}"
        self.name_counter += 1
        return name

    # return (str, str) with lhs being stmts, and rhs being expression
    def visit(self, node, local=True):
        if isinstance(node, little_cpp.PackedCall):
            res = self.visit_packed_call(node)
        elif isinstance(node, little_cpp.CPPFunction):
            res = self.visit_cpp_function(node, local)
        elif isinstance(node, little_cpp.Decl):
            res = self.visit_decl(node)
        elif isinstance(node, little_cpp.Invoke):
            res = self.visit_invoke(node)
        elif isinstance(node, relay.Var):
            res = ("", self.name_map[node])
        elif isinstance(node, relay.GlobalVar):
            res = self.visit_global_var(node)
        else:
            raise Exception(str(node))
        return res

    def visit_type(self, node):
        if isinstance(node, relay.TensorType):
            res = "NDArray"
        else:
            raise Exception(str(node))
        return res

    def visit_global_var(self, gv):
        if gv not in self.declare_map:
            lhs, rhs = self.visit(self.gv_map[gv], local=False)
            assert lhs == ""
            self.declare_map[gv] = rhs
        return ("", self.declare_map[gv])

    def visit_invoke(self, invoke):
        decl_str = ""
        args_str = ""
        for i, arg in enumerate(invoke.args):
            assert isinstance(arg, relay.Var)
            lhs, rhs = self.visit(arg)
            decl_str += lhs
            args_str += rhs
            if i != len(invoke.args) - 1:
                args_str += ", "

        func = self.visit(invoke.call)
        return (decl_str, f"{func}({args_str})")

    def visit_decl(self, decl):
        source = ""
        for var, value in decl.bindings:
            local_name = self.fresh_local_name(var)
            self.name_map[var] = local_name
            value_str = self.visit(value)
            source += f"auto {local_name} = {value_str};\n"
        lhs, rhs = self.visit(decl.body)
        source += lhs
        return (source, rhs)

    def visit_packed_call(self, call):
        decl_str = ""
        args_str = ""
        end = len(call.args) - 1
        for i, arg in enumerate(call.args):
            lhs, rhs = self.visit(arg)
            decl_str += lhs
            args_str += rhs
            if i != end:
                args_str += ", "

        out_name = self.fresh_local_name()
        return (f"""
            const PackedFunc *pf = runtime::Registry::Get("{call.name}");
            CHECK(pf);
            NDArray {out_name} = NDArray::Empty({{}}, dtype_f32, context);
            (*pf)({out_name}, out);
        """, out_name)

    def visit_cpp_function(self, func, local):
        param_str = ""

        end = len(func.params) - 1
        for i, param in enumerate(func.params):
            pname = f"param{i}"
            self.name_map[param] = pname
            param_str += f"const {self.visit_type(param.type_annotation)}& {pname}"
            if i != end:
                param_str += ", "

        lhs, rhs = self.visit(func.body)
        body = lhs + f"""return {rhs};"""

        if local:
            return ("", f"""[=]({param_str}) {{
                {body}
            }}
            """)
        else:
            name = self.fresh_global_name()
            print(func)
            print(func.ret_type)
            self.declare += f"""
            NDArray {name}({param_str}) {{
                {body}
            }}
            """
            return ("", name)

    def mk_register_api(self, name: str, func: little_cpp.CPPFunction) -> str:
        assert isinstance(func, little_cpp.CPPFunction)
        lhs, fname = self.visit(func, False)
        assert lhs == ""
        source = self.declare

        args = ""
        end = len(func.params) - 1
        for i in range(len(func.params)):
            args += f"args[{i}]"
            if i != end:
                args += ", "

        source += f"""
        TVM_REGISTER_API("{name}")
        .set_body([](TVMArgs args, TVMRetValue* ret) {{
            *ret = {fname}({args});
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

def to_source(gv_map, name, program) -> str:
    assert isinstance(program, little_cpp.CPPFunction)
    convert = ToSource(gv_map)
    return mk_file(convert.mk_register_api(name, program))
