import torch
import tvm
from tvm import relay
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay import testing, create_executor
from tvm.relay.prelude import Prelude

from .treelstm import TreeLSTM, LSTMCell
import aot

class RoseTree:
    def __init__(self, head, children):
        self.head = head
        self.children = children

    def __str__(self):
        return "Tree(" + str(self.head) + ", " + str(self.children) + ")"

    def __repr__(self):
        return self.__str__()

    def fmap(self, f):
        return RoseTree(f(self.head), [x.fmap(f) for x in self.children])

    def size(self):
        return 1 + sum([x.size() for x in self.children])


def make_nat(p, n):
    if n != 0:
        return ConstructorValue(p.s.tag, [make_nat(n - 1)], None)
    else:
        return ConstructorValue(p.z.tag, [], None)


# creates relay list from a list
def from_list(p, l, t):
    if len(l) == 0:
        return ConstructorValue(p.nil.tag, [], None)
    else:
        return ConstructorValue(p.cons.tag, [l[0], from_list(p, l[1:], t)], None)

# convert tensors
def pytorch_to_relay(tensor):
    #print(tensor.shape)
    return relay.const(tensor.detach().cpu().numpy().reshape((1, 300)), dtype='float32').data


def from_tree(p, rt, t):
    return ConstructorValue(p.rose.tag,
                            [rt.head,
                             from_list(p, [from_tree(p, x, t) for x in rt.children], t)], None)


def forward(tree, inputs):
    children = [forward(x, inputs) for x in tree.children]
    return RoseTree(inputs[tree.idx], children)


def tree_to_dict(t):
    assert isinstance(t, ConstructorValue)
    ret = {}
    ret['member'] = t.fields[0]
    ret['children'] = []
    for subtree in to_list(t.fields[1]):
        l = tree_to_dict(subtree)
        ret['children'].append(l)
    return ret


def initialize_tlstm(input_size, memory_size):
    tlstm = TreeLSTM(input_size=input_size, memory_size=memory_size)
    return tlstm, tlstm.mod, tlstm.p
