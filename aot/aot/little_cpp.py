from typing import Any, Optional
import attr

class LittleCppNode:
    pass

@attr.s(auto_attribs=True)
class PackedCall(LittleCppNode):
    name: str
    arity: int
    args: Any
    output_type: Any

@attr.s(auto_attribs=True)
class CPPFunction(LittleCppNode):
    params: Any
    body: Any
    ret_type: Any
    name: Optional[str] = None

