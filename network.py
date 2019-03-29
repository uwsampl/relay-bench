import numpy as np
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay import create_executor, Module
from tvm.relay.backend.interpreter import TensorValue
from tvm.relay.prelude import Prelude
import aot
import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self):
        key = self.last()
        self.discard(key)
        return key

    def last(self):
        return self.end[1][0]

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

def initialize(param):
    ty = param.type_annotation
    shape = [int(i) for i in ty.shape]
    return np.random.normal(0, 1, shape).astype('float32')

class Network:
    NetworkStack = OrderedSet()

    def __init__(self, *args, name="f", mod = None):
        if mod is None:
            mod = Module()

        self.mod = mod
        self.prelude = Prelude(self.mod)
        self.inputs = []
        self.weights = OrderedSet()
        self.sub_network = OrderedSet()
        self.f = relay.GlobalVar(name)
        self.recurse = relay.Var("recurse")
        self.use_recurse = False
        body = self.build(*args)
        if self.use_recurse:
            body = relay.Let(recurse, self(*inputs), body)
        self.mod = relay.Function(self.inputs + self.all_weights(), body)

    def build(*args):
        try:
            NetworkStack.add()
            ret = build_impl(*args)
            NetworkStack.pop()
            NetworkStack.last().sub_network.add(self)
            return ret
        except:
            NetworkStack.pop()
            raise

    def build_impl(*args):
        raise NotImplemented

    def add_weight(self, w):
        assert isinstance(w, relay.Var)
        self.weights.add(w)

    def add_input(self, i):
        assert isinstance(i, relay.Var)
        self.inputs.add(i)

    def all_weights(self):
        return list(weights) + [w for n in self.sub_network for w in n.all_weights()]

    def __call__(self, *inputs):
        if self in NetworkStack:
            return recurse(*inputs)
        else:
            return self.f(*(inputs + all_weights()))

def linear(self, input_size, output_size, x, name=""):
    weight = self.add_param(f'{name}linear_weight', shape=(output_size, input_size))
    bias = self.add_param(f'{name}linear_bias', shape=(output_size,))
    return op.add(op.nn.dense(x, weight), bias)
