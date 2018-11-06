from . import little_cpp

#  PackedFunc *pf = reinterpret_cast<PackedFunc*>({jit_func.handle.value});
#         CHECK(pf);
#         (*pf)({args});

class ToSource:
    def __init__():
        self.name_counter = 0
        self.source_content = ""

    def fresh_global_name():
        pass

    def fresh_local_name():
        pass

    def visit_packed_call(self, call):
        pass

    def visit_cpp_function(self, call):
        pass

    def mk_register_api(self, func: little_cpp.CPPFunction) -> str:
        assert isinstance(func, CPPFunction)

        self.visit_cpp_function(func)

        args = ""
        for i in range(arity):
            args += f"args[{i}]"
            if i != arity - 1:
                args += ", "

        register = f"""
        TVM_REGISTER_API("{name}")
        .set_body([](TVMArgs args, TVMRetValue* ret) {{
            PackedFunc *pf = reinterpret_cast<PackedFunc*>({jit_func.handle.value});
            CHECK(pf);
            (*pf)({args});
        }});
        """

def to_source(program) -> str:
    assert isinstance(program, little_cpp.CPPFunction)
    convert = ToSource()
    convert.mk_register_api(program)
    return convert.source_content
