from . import little_cpp
from tvm import relay
from tvm.relay import _module
from tvm.relay.prelude import Prelude

class ExprWithStmt:
    def __init__(self, expr, stmt=""):
        assert isinstance(expr, str)
        assert isinstance(stmt, str)
        assert "ExprWithStmt" not in expr
        assert "ExprWithStmt" not in stmt
        self.expr = expr
        self.stmt = stmt

    def __str__(self):
        return f"ExprWithStmt({self.expr}, {self.stmt})"

    def __repr__(self):
        return self.__str__()

class ToSource:
    def __init__(self, gv_map):
        self.gv_map = gv_map
        self.name_counter = 0
        self.source_content = ""
        self.name_map = {}
        self.local = True
        self.declare = ""
        self.declare_map = {}
        self.input_const = []

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
    def visit(self, node, local=True, name=None):
        if isinstance(node, little_cpp.PackedCall):
            res = self.visit_packed_call(node)
        elif isinstance(node, little_cpp.CPPFunction):
            res = self.visit_cpp_function(node, local, name)
        elif isinstance(node, little_cpp.Decl):
            res = self.visit_decl(node)
        elif isinstance(node, little_cpp.Invoke):
            res = self.visit_invoke(node)
        elif isinstance(node, relay.Var):
            res = ExprWithStmt(self.name_map[node])
        elif isinstance(node, relay.GlobalVar):
            res = self.visit_global_var(node)
        elif isinstance(node, relay.Constant):
            res = self.visit_constant(node)
        elif isinstance(node, little_cpp.CPPIf):
            res = self.visit_if(node)
        elif isinstance(node, little_cpp.CPPTuple):
            res = self.visit_tuple(node)
        else:
            raise Exception(str(node))
        assert isinstance(res, ExprWithStmt)
        return res

    def visit_tuple(self, node):
        expr = []
        stmt_str = ""
        for x in node.fields:
            vx = self.visit(x)
            expr.append(vx.expr)
            stmt_str += vx.stmt
        return ExprWithStmt(f"TupleValueNode::make({{{inter(expr)}}})", stmt_str)

    def visit_if(self, node):
        vc = self.visit(node.cond)
        vt = self.visit(node.true_branch)
        vf = self.visit(node.false_branch)
        ret_name = self.fresh_local_name()
        stmt = f"{self.visit_type(node.relay_type)} {ret_name};"
        stmt += f"""
        {vc.stmt}
        if (NDToBool({vc.expr}->data)) {{
          {vt.stmt}
          {ret_name} = {vt.expr};
        }} else {{
          {vf.stmt}
          {ret_name} = {vf.expr};
        }}
        """
        return ExprWithStmt(ret_name, stmt)

    def visit_type(self, node):
        if isinstance(node, relay.TensorType):
            res = "TensorValue"
        elif isinstance(node, relay.TupleType):
            res = "TupleValue"
        else:
            raise Exception(str(node))
        return res

    def visit_constant(self, const):
        if const not in self.declare_map:
            name = self.fresh_global_name()
            self.declare_map[const] = name
            self.declare += f"TensorValue {name};\n"
            self.input_const.append((name, const))
        return ExprWithStmt(self.declare_map[const])

    def visit_global_var(self, gv):
        if gv not in self.declare_map:
            name = self.fresh_global_name()
            self.declare_map[gv] = name
            vgv = self.visit(self.gv_map[gv], local=False, name=name)
            assert vgv.stmt == ""
            assert vgv.expr == name
        return ExprWithStmt(self.declare_map[gv])

    def visit_invoke(self, invoke):
        decl_str = ""
        args_str = ""
        for i, arg in enumerate(invoke.args):
            assert isinstance(arg, relay.Var)
            va = self.visit(arg)
            decl_str += va.stmt
            args_str += va.expr
            if i != len(invoke.args) - 1:
                args_str += ", "

        func = self.visit(invoke.call)
        return ExprWithStmt(f"{func.expr}({args_str})", decl_str + func.stmt)

    def visit_decl(self, decl):
        source = ""
        for var, value in decl.bindings:
            local_name = self.fresh_local_name(var)
            self.name_map[var] = local_name
            vv = self.visit(value)
            source += vv.stmt
            source += f"auto {local_name} = {vv.expr};\n"
        vb = self.visit(decl.body)
        source += vb.stmt
        return ExprWithStmt(vb.expr, source)

    def empty_nd(self, tt):
        assert isinstance(tt, relay.ty.TensorType)
        if tt.dtype == 'int32':
            return 'dtype_i32'
        elif tt.dtype == 'float32':
            return 'dtype_f32'
        elif tt.dtype == 'bool':
            return 'dtype_u1'
        raise Exception("unknown tensor dtype: " + str(tt))

    def visit_packed_call(self, call):
        decl_str = ""
        args_str = ""
        if call.args_is_tuple:
            assert len(call.args) == 1
            va = self.visit(call.args[0])
            decl_str += va.stmt
            tuple_name = self.fresh_local_name();
            decl_str += f"TupleValue {tuple_name} = {va.expr};\n"
            end = call.arity - 2
            for i in range(end + 1):
                args_str += f"{tuple_name}->fields[{i}]"
                if i != end:
                    args_str += ", "
            print(args_str)
        else:
            end = call.arity - 2
            for i, arg in enumerate(call.args):
                va = self.visit(arg)
                decl_str += va.stmt
                args_str += f"{va.expr}->data"
                if i != end:
                    args_str += ", "

        out_name = self.fresh_local_name()
        return ExprWithStmt(out_name, f"""
            {decl_str}
            const PackedFunc *pf = runtime::Registry::Get("{call.name}");
            CHECK(pf);
            TensorValue {out_name} = TensorValueNode::make(NDArray::Empty({{}}, {self.empty_nd(call.output_type)}, context));
            (*pf)({args_str}, {out_name}->data);
        """)

    def visit_cpp_function(self, func, local, name):
        param_str = ""

        end = len(func.params) - 1
        for i, param in enumerate(func.params):
            pname = f"param{i}"
            self.name_map[param] = pname
            param_str += f"const {self.visit_type(param.type_annotation)}& {pname}"
            if i != end:
                param_str += ", "

        vb = self.visit(func.body)
        body = vb.stmt + f"""return {vb.expr};"""

        if local:
            return ExprWithStmt(f"""[=]({param_str}) {{
                {body}
            }}
            """)
        else:
            if name is None:
                name = self.fresh_global_name()
            self.declare += f"""
            {self.visit_type(func.ret_type)} {name}({param_str}) {{
                {body}
            }}
            """
            return ExprWithStmt(name)

    def mk_register_api(self, name: str, func: little_cpp.CPPFunction) -> str:
        assert isinstance(func, little_cpp.CPPFunction)
        vf = self.visit(func, False)
        assert vf.stmt == ""
        source = self.declare

        args = ""
        end = len(func.params) - 1
        init = ""
        for i, (input_name, _) in enumerate(self.input_const):
            init += f"{input_name} = args[{i}];\n"
        for i in range(len(func.params)):
            args += f"args[{i+len(self.input_const)}]"
            if i != end:
                args += ", "

        source += f"""
        TVM_REGISTER_API("{name}")
        .set_body([](TVMArgs args, TVMRetValue* ret) {{
            {init}
            *ret = {vf.expr}({args});
        }});
        """
        print(source)
        return source

def inter(strs, sep=", "):
    ret = ""
    for i in range(len(strs)):
        ret += strs[i]
        if i != len(strs) - 1:
            ret += sep
    return ret

def mk_file(body):
    return f"""
    #include <tvm/tvm.h>
    #include <tvm/api_registry.h>
    #include <tvm/relay/interpreter.h>
    #include <iostream>

    using namespace tvm;
    using namespace runtime;
    using namespace relay;

    static DLDataType dtype_f32 = DLDataType {{ .code = DLDataTypeCode::kDLFloat, .bits = 32, .lanes = 1 }};
    static DLDataType dtype_u32 = DLDataType {{ .code = DLDataTypeCode::kDLUInt, .bits = 32, .lanes = 1 }};
    static DLDataType dtype_u1 = DLDataType {{ .code = DLDataTypeCode::kDLUInt, .bits = 1, .lanes = 1 }};
    static DLDataType dtype_i32 = DLDataType {{ .code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1 }};
    static DLContext context = DLContext {{ .device_type = DLDeviceType::kDLCPU, . device_id = 0 }};
    bool NDToBool (const NDArray& nd) {{
      DLContext cpu_ctx;
      cpu_ctx.device_type = kDLCPU;
      cpu_ctx.device_id = 0;
      NDArray cpu_array = nd.CopyTo(cpu_ctx);
      CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
      return reinterpret_cast<uint8_t*>(cpu_array->data)[0];
    }}
    {body}
    """

def to_source(mod, gv_map, name, program) -> str:
    assert isinstance(program, little_cpp.CPPFunction)
    convert = ToSource(gv_map)
    ret = mk_file(convert.mk_register_api(name, program))
    return [value.data for name, value in convert.input_const], ret
