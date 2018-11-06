from typing import Any
import attr

class LittleCppNode:
    pass

class PackedCall(LittleCppNode):
    packed_func: Any
    arity: int
    args: Any

class Function(LittleCppNode):
    params: Any
    body: Any

