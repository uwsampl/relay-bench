from io import open
import glob
import os
import random
import unicodedata
import string
import time
import math
from rnn import language_data as data
import numpy as np
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay import create_executor, Module
from tvm.relay.prelude import Prelude

def linear(input_size, output_size, x):
    weight = relay.var('linear_weight', shape=(input_size, output_size))
    bias = relay.var('linear_bias', shape=(output_size,))
    return op.add(op.nn.dense(x, weight), bias), weight, bias

mod = Module()
p = Prelude(mod)
ctx = tvm.context("llvm", 0)
intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

def init(shape):
    return np.random.normal(0, 1, shape).astype('float32')

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.fwd = relay.GlobalVar('fwd')
        self.hidden = init((1, hidden_size))
        category = relay.var('category', shape=(1, data.N_CATEGORIES))
        inp = relay.var('input', shape=(1, input_size))
        hidden_var = relay.var('hidden', shape=(1, hidden_size))
        combined = op.concatenate2(op.concatenate2(category, inp, axis=1), hidden_var, axis=1)
        hidden, self.w0_var, self.b0_var = linear(data.N_CATEGORIES + input_size + hidden_size, hidden_size, combined)
        output, self.w1_var, self.b1_var = linear(data.N_CATEGORIES + input_size + hidden_size, output_size, combined)
        output_combined = op.concatenate2(hidden, output, axis=1)
        output, self.w2_var, self.b2_var = linear(hidden_size + output_size, output_size, output_combined)
        # output = op.nn.dropout(output, 0.1) #attributes has not been registered
        output = op.nn.log_softmax(output, axis=1)
        body = relay.Tuple([output, hidden, op.argmax(output)])
        assert len(relay.ir_pass.free_vars(body)) == 9
        para = [category, inp, hidden_var, self.w0_var, self.b0_var, self.w1_var, self.b1_var, self.w2_var, self.b2_var]
        mod[self.fwd] = relay.Function(para, body)
        self.w0 = init((data.N_CATEGORIES + input_size + hidden_size, hidden_size))
        self.b0 = init(hidden_size)
        self.w1 = init((data.N_CATEGORIES + input_size + hidden_size, output_size))
        self.b1 = init(output_size)
        self.w2 = init((hidden_size + output_size, output_size))
        self.b2 = init(output_size)
        self.forward = intrp.evaluate(self.fwd)

    def __call__(self, category, input, hidden):
        return self.forward(category, input, hidden, self.w0, self.b0, self.w1, self.b1, self.w2, self.b2)

