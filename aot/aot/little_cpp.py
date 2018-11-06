from typing import Any
import attr

class LittleCppNode:
    pass

@attr.s(auto_attribs=True)
class PackedCall(LittleCppNode):
    packed_func: Any
    arity: int
    args: Any

@attr.s(auto_attribs=True)
class CPPFunction(LittleCppNode):

    params: Any
    body: Any

