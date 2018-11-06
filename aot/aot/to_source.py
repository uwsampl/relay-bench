from . import little_cpp

#  PackedFunc *pf = reinterpret_cast<PackedFunc*>({jit_func.handle.value});
#         CHECK(pf);
#         (*pf)({args});

class ToSource:
    def __init__(self):
        self.name_counter = 0
        self.source_content = ""
        self.name_map = {}
        self.cont = None

    def fresh_global_name(self):
        name = f"global{self.name_counter}"
        self.name_counter += 1
        return name

    def fresh_local_name():
        pass

    def do_cont():
        return "return"

    def visit_packed_call(self, call):
        args = ""
        end = len(call.args) - 1
        for i, arg in enumerate(call.args):
            args += self.name_map[arg]
            if i != end:
                args += ", "

        return f"""
            const PackedFunc *pf = runtime::Registry::Get("{call.name}");
            CHECK(pf);
            NDArray out = NDArray::Empty({{}}, dtype_f32, context);
            (*pf)({args}, out);
            return out;
        """

    def visit_cpp_function(self, func):
        name = func.name = self.fresh_global_name()
        assert isinstance(func.body, little_cpp.PackedCall)
        param_str = ""
        args = ""

        end = len(func.params) - 1
        for i, param in enumerate(func.params):
            pname = f"param{i}"
            self.name_map[param] = pname
            param_str += f"const NDArray& {pname}"
            if i != end:
                param_str += ", "

        body = self.visit_packed_call(func.body)

        func = f"""
        NDArray {name}({param_str}) {{
            {body}
        }}
        """
        return func

    def mk_register_api(self, name: str, func: little_cpp.CPPFunction) -> str:
        assert isinstance(func, little_cpp.CPPFunction)
        source = ""
        source += self.visit_cpp_function(func)

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
