from io import open
import glob
import os
import random
import unicodedata
import string
import time
import math
from rnn.language_data import N_CATEGORIES, N_LETTERS
from rnn.relay.util import categoryTensor, inputTensor
import numpy as np
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay import create_executor, Module
from tvm.relay.backend.interpreter import TensorValue
from tvm.relay.prelude import Prelude
from aot import aot

def initialize(param):
    ty = param.type_annotation
    shape = [int(i) for i in ty.shape]
    return TensorValue(
        np.random.normal(0, 1, shape).astype('float32'))

class Network:
    def add_param(self, name="", shape=()):
        x = relay.var(name, shape=shape)
        self.parameters.append((x, initialize(x)))
        return x

    def linear(self, input_size, output_size, x, name=""):
        weight = self.add_param(f'{name}linear_weight', shape=(output_size, input_size))
        bias = self.add_param(f'{name}linear_bias', shape=(output_size,))
        return op.add(op.nn.dense(x, weight), bias)

    def __init__(self, *args):
        self.mod = Module()
        self.prelude = Prelude(self.mod)
        self.context = tvm.cpu(0)
        self.target = tvm.target.create('llvm')
        self.executor = create_executor(mod=self.mod, ctx=self.context)
        self.parameters = []
        self.forward_var = relay.GlobalVar('forward_var')

        # Set up forward pass.
        inputs, body, ret_type = self.compute(*args)
        self.inputs = inputs

        forward_compute = relay.Function(inputs + list([p[0] for p in self.parameters]), body, ret_type)
        self.mod[self.forward_var] = forward_compute
        self.forward = aot.compile(self.mod, self.forward_var)
        #self.forward = self.executor.static_evaluate(self.forward_var)
        self.args = [None] * len(inputs) + list([p[1] for p in self.parameters])

    def __call__(self, *inputs):
        for i, inp in enumerate(inputs):
            self.args[i] = inp

        return self.forward(*self.args)

    def recurse(self, *inputs):
        return self.forward_var(*inputs, *[p[0] for p in self.parameters])

class RNNCellOnly(Network):
    def compute(self, input_size, hidden_size, output_size):
        self.category_var = category = relay.var('category', shape=(1, N_CATEGORIES))
        self.input_var = inp = relay.var('input', shape=(1, input_size))
        self.hidden_var = hidden = relay.var('hidden', shape=(1, hidden_size))
        self.hidden = initialize(self.hidden_var)
        combined = op.concatenate2(op.concatenate2(category, inp, axis=1), hidden, axis=1)
        hidden = self.linear(N_CATEGORIES + input_size + hidden_size, hidden_size, combined, name='i2h')
        output = self.linear(N_CATEGORIES + input_size + hidden_size, output_size, combined, name='i2o')
        output_combined = op.concatenate2(hidden, output, axis=1)
        output = self.linear(hidden_size + output_size, output_size, output_combined, name='o2o')
        #output = op.nn.dropout(output, 0.1) #dropout isnt simplified, commented out for now
        output = op.nn.log_softmax(output, axis=1)
        return [self.category_var, self.input_var, self.hidden_var], relay.Tuple([output, hidden]), None

    def warm(self):
        self(initialize(self.category_var), initialize(self.input_var), initialize(self.hidden_var))

def copy_var(v):
    return relay.Var(name_hint=v.name_hint, type_annotation=v.type_annotation)

class RNNLoop(Network):
    def compute(self, input_size, hidden_size, output_size):
        self.category_var = category = relay.var('category', shape=(1, N_CATEGORIES))
        self.inp_topi_var = inp_topi = relay.var('input', shape=(), dtype='int32')
        self.hidden_var = hidden = relay.var('hidden', shape=(1, hidden_size))
        n_letter = relay.const(N_LETTERS)
        one_diag = relay.const(np.diag(np.ones(58)).astype('float32'))
        boxed_one = relay.const(np.array([1]).astype('int32'))
        inp = op.take(one_diag, op.multiply(boxed_one, inp_topi), axis=0)
        combined = op.concatenate2(op.concatenate2(category, inp, axis=1), hidden, axis=1)
        hidden = self.linear(N_CATEGORIES + input_size + hidden_size, hidden_size, combined, name='i2h')
        output = self.linear(N_CATEGORIES + input_size + hidden_size, output_size, combined, name='i2o')
        output_combined = op.concatenate2(hidden, output, axis=1)
        output = self.linear(hidden_size + output_size, output_size, output_combined, name='o2o')
        # output = op.nn.dropout(output, 0.1) #attributes has not been registered
        output = op.nn.log_softmax(output, axis=1)
        topi = op.argmax(output)
        body = relay.Tuple([output,
                            hidden,
                            topi,
                            op.equal(topi, op.subtract(n_letter, relay.const(1)))])
        fwd_para = [self.category_var, self.inp_topi_var, self.hidden_var]
        fwd_func = relay.Function(fwd_para, body)
        self.fwd = relay.Var('fwd')

        max = relay.var('max', shape=(), dtype='int32')
        inp_para = [max] + [copy_var(v) for v in fwd_para]
        fwd_res = self.fwd(*inp_para[1:])
        else_else_branch = self.prelude.cons(fwd_res[2], self.recurse(op.subtract(max, relay.const(1)), category, fwd_res[2], fwd_res[1]))
        else_branch = relay.If(fwd_res[3], self.prelude.nil(), else_else_branch)
        body = relay.If(op.equal(max, relay.const(0)), self.prelude.nil(), else_branch)
        return inp_para, relay.Let(self.fwd, fwd_func, body), None

#     def woosh(self, l):
#         if l.con.name_hint == 'cons':
#             return [np.asscalar(l.fields[0].data.asnumpy())] + self.woosh(l.fields[1])
#         else:
#             assert l.con.name_hint == 'nil'
#             return []

#     def sample(self, category, start_letter='A'):
#         category_tensor = categoryTensor(category)
#         input = data.letter_to_topi(start_letter)
#         hidden = self.hidden
#         output = self.loop_forward(20,
#                                    category_tensor,
#                                    input,
#                                    hidden,
#                                    self.w0,
#                                    self.b0,
#                                    self.w1,
#                                    self.b1,
#                                    self.w2,
#                                    self.b2)
#         output_name = ''
#         for x in [data.letter_to_topi(start_letter)] + self.woosh(output):
#             output_name += data.topi_to_letter(x)
#         return output_name

#     def samples(self, category, start_letters='ABC'):
#         for start_letter in start_letters:
#             print(self.sample(category, start_letter))
