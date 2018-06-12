import relay.make as mk
from relay.frontend import get_env
from relay.frontend import relay
from relay.typing import Tensor, Float
import relay.eval as re

@relay
def foo() -> Tensor[Float, (2)]:
    return [1.0, 1.0]

env = get_env()
hello_world = env.global_id("hello_world")
ty = None
shape = mk.ShapeSeq([mk.ShapeSingleton(2)])
empty_fn_type = mk.TypeArrow(mk.ProductType([]), mk.TensorType(mk.FloatType(32), shape))
body = mk.Function([], empty_fn_type, mk.Call(env.intrinsic_id("softmax"), [mk.Call(env.global_id("foo"), [])]))

defn = mk.Defn(hello_world, ty, body)
env.add(defn)
re.invoke(env, defn.id, [])

import pdb; pdb.set_trace()
