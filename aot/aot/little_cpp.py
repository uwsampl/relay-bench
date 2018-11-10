from tvm.relay import Var, TypeVar
from typing import Any, Optional, List, Tuple
import attr

#remember: template must not have 0 param.

class LittleCppNode:
    pass

@attr.s(auto_attribs=True)
class Decl(LittleCppNode):
    bindings: List[Tuple[Var, LittleCppNode]]
    body: LittleCppNode

@attr.s(auto_attribs=True)
class PackedCall(LittleCppNode):
    name: str
    arity: int
    args: Any
    output_type: Any

@attr.s(auto_attribs=True)
class Invoke(LittleCppNode):
    call: Any
    args: Any

@attr.s(auto_attribs=True)
class CPPFunction(LittleCppNode):
    params: List[Var]
    body: Any
    ret_type: Any
    name: Optional[str] = None
