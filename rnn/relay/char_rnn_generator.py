from io import open
import glob
import os
import random
import unicodedata
import string
import time
import math
from rnn import language_data as data
from rnn.relay.util import categoryTensor, inputTensor
import numpy as np
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay import create_executor, Module
from tvm.relay.backend.interpreter import TensorValue
from tvm.relay.prelude import Prelude

def linear(input_size, output_size, x, name=None):
    if name is None:
        name = ""

    weight = relay.var(f'{name}linear_weight', shape=(output_size, input_size))
    bias = relay.var(f'{name}linear_bias', shape=(output_size,))
    return op.add(op.nn.dense(x, weight), bias)

def initialize(param):
    ty = param.type_annotation
    shape = [int(i) for i in ty.shape]
    return TensorValue(
        np.random.normal(0, 1, shape).astype('float32'))

class RNNCellOnly:
    def __init__(self, input_size, hidden_size, output_size):
        mod = Module()
        ctx = tvm.context("llvm", 0)
        intrp = create_executor(mod=mod, ctx=ctx, target="llvm")
        self.parameters = {}

        self.category_var = category = relay.var('category', shape=(1, data.N_CATEGORIES))
        self.input_var = inp = relay.var('input', shape=(1, input_size))
        self.hidden_var = hidden = relay.var('hidden', shape=(1, hidden_size))

        combined = op.concatenate([category, inp, hidden], axis=1)
        hidden = linear(data.N_CATEGORIES + input_size + hidden_size, hidden_size, combined, name='i2h')
        output = linear(data.N_CATEGORIES + input_size + hidden_size, output_size, combined, name='i2o')
        output_combined = op.concatenate([hidden, output], axis=1)
        output = linear(hidden_size + output_size, output_size, output_combined, name='o2o')
        # output = op.nn.dropout(output, 0.1) #attributes has not been registered
        output = op.nn.log_softmax(output, axis=1)
        body = relay.Tuple([output, hidden])
        free_vars = relay.ir_pass.free_vars(body)
        for param in free_vars[3:]:
            self.parameters[param] = initialize(param)
        self.hidden = initialize(free_vars[2])
        assert len(relay.ir_pass.free_vars(body)) == 9
        self.fwd = relay.Function(free_vars, body)
        self.forward = intrp.static_evaluate(self.fwd)

    def warm(self):
        self.forward(initialize(self.category_var), initialize(self.input_var), initialize(self.hidden_var), *self.parameters.values())

    def __call__(self, category, input, hidden):
        return self.forward(category, input, hidden, *self.parameters.values())

class RNNLoop:
    def __init__(self, input_size, hidden_size, output_size):
        mod = Module()
        p = Prelude(mod)
        ctx = tvm.context("llvm", 0)
        intrp = create_executor(mod=mod, ctx=ctx, target="llvm")
        self.parameters = {}

        self.category_var = category = relay.var('category', shape=(1, data.N_CATEGORIES))
        self.inp_topi_var = inp_topi = relay.var('input', shape=(), dtype='int32')
        self.hidden_var = hidden = relay.var('hidden', shape=(1, hidden_size))
        n_letter = relay.const(data.N_LETTERS)
        one_diag = relay.const(np.diag(np.ones(58)).astype('float32'))
        boxed_one = relay.const(np.array([1]).astype('int32'))
        inp = op.take(one_diag, op.multiply(boxed_one, inp_topi), axis=0)
        combined = op.concatenate([category, inp, hidden], axis=1)
        combined = op.concatenate([category, inp, hidden], axis=1)
        hidden = linear(data.N_CATEGORIES + input_size + hidden_size, hidden_size, combined, name='i2h')
        output = linear(data.N_CATEGORIES + input_size + hidden_size, output_size, combined, name='i2o')
        output_combined = op.concatenate([hidden, output], axis=1)
        output = linear(hidden_size + output_size, output_size, output_combined, name='o2o')
        # output = op.nn.dropout(output, 0.1) #attributes has not been registered
        output = op.nn.log_softmax(output, axis=1)
        body = relay.Tuple([output, hidden])
        free_vars = relay.ir_pass.free_vars(body)
        for param in free_vars[3:]:
            self.parameters[param] = initialize(param)
        self.hidden = initialize(free_vars[2])
        assert len(relay.ir_pass.free_vars(body)) == 9
        self.fwd = relay.Function(free_vars, body)
        self.forward = intrp.static_evaluate(self.fwd)

#         self.fwd = relay.GlobalVar('fwd')
#         topi = op.argmax(output)
#         body = relay.Tuple([output,
#                             hidden,
#                             topi,
#                             op.equal(topi, op.subtract(n_letter, relay.const(1)))])
#         assert len(relay.ir_pass.free_vars(body)) == 9
#         inp_para = [category, inp_topi, hidden_var]
#         weight_para = [self.w0_var, self.b0_var, self.w1_var, self.b1_var, self.w2_var, self.b2_var]
#         para = inp_para + weight_para
#         self.w0 = init((data.N_CATEGORIES + input_size + hidden_size, hidden_size))
#         self.b0 = init(hidden_size)
#         self.w1 = init((data.N_CATEGORIES + input_size + hidden_size, output_size))
#         self.b1 = init(output_size)
#         self.w2 = init((hidden_size + output_size, output_size))
#         self.b2 = init(output_size)
#         mod[self.fwd] = relay.Function(para, body)
#         self.forward = intrp.static_evaluate(self.fwd)

#         self.loop_fwd = relay.GlobalVar('loop_fwd')
#         max = relay.var('max', shape=(), dtype='int32')
#         loop_para = [max] + para
#         fwd_res = self.fwd(*para)
#         else_else_branch = p.cons(fwd_res[2], self.loop_fwd(op.subtract(max, relay.const(1)), category, fwd_res[2], fwd_res[1], *weight_para))
#         else_branch = relay.If(fwd_res[3], p.nil(), else_else_branch)
#         body = relay.If(op.equal(max, relay.const(0)), p.nil(), else_branch)
#         mod[self.loop_fwd] = relay.Function(loop_para, body)
#         print(mod[self.loop_fwd].checked_type)
#         self.loop_forward = intrp.static_evaluate(self.loop_fwd)

#     def __call__(self, category, input, hidden):
#         return self.forward(category, input, hidden, self.w0, self.b0, self.w1, self.b1, self.w2, self.b2)

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
