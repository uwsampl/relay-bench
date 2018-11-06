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
from tvm.relay.prelude import Prelude

def linear(input_size, output_size, x):
    weight = relay.var('linear_weight', shape=(input_size, output_size))
    bias = relay.var('linear_bias', shape=(output_size,))
    return op.add(op.nn.dense(x, weight), bias), weight, bias

def init(shape):
    return np.random.normal(0, 1, shape).astype('float32')

def letter_to_topi(letter):
    return data.ALL_LETTERS.index(letter)

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        mod = Module()
        p = Prelude(mod)
        ctx = tvm.context("llvm", 0)
        intrp = create_executor(mod=mod, ctx=ctx, target="llvm")
        self.fwd = relay.GlobalVar('fwd')
        self.hidden = init((1, hidden_size))
        category = relay.var('category', shape=(1, data.N_CATEGORIES))
        n_letter = relay.const(data.N_LETTERS)
        one_diag = relay.const(np.diag(np.ones(58)).astype('float32'))
        boxed_one = relay.const(np.array([1]).astype('int32'))
        inp = relay.var('input', shape=(1, input_size))
        hidden_var = relay.var('hidden', shape=(1, hidden_size))
        combined = op.concatenate([category, inp, hidden_var], axis=1)
        hidden, self.w0_var, self.b0_var = linear(data.N_CATEGORIES + input_size + hidden_size, hidden_size, combined)
        output, self.w1_var, self.b1_var = linear(data.N_CATEGORIES + input_size + hidden_size, output_size, combined)
        output_combined = op.concatenate([hidden, output], axis=1)
        output, self.w2_var, self.b2_var = linear(hidden_size + output_size, output_size, output_combined)
        # output = op.nn.dropout(output, 0.1) #attributes has not been registered
        output = op.nn.log_softmax(output, axis=1)
        topi = op.argmax(output)
        body = relay.Tuple([output,
                            hidden,
                            topi,
                            op.equal(topi, op.subtract(n_letter, relay.const(1))),
                            op.take(one_diag, op.multiply(boxed_one, topi), axis=0)])
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

    def sample(self, category, start_letter='A'):
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = self.hidden
        output_name = start_letter
        output_topi = [letter_to_topi(start_letter)]
        for i in range(data.MAX_LENGTH):
            output, hidden, topi, b, input = self.forward(category_tensor,
                                                          input,
                                                          hidden,
                                                          self.w0,
                                                          self.b0,
                                                          self.w1,
                                                          self.b1,
                                                          self.w2,
                                                          self.b2)
            if b.data.asnumpy():
                break
            else:
                topi = topi.data.asnumpy()
                letter = data.ALL_LETTERS[topi]
                output_topi.append(topi)
                output_name += letter
        output_name_new = ''
        for x in output_topi:
            output_name_new += data.ALL_LETTERS[x]
        assert output_name_new == output_name
        return output_name

    def samples(self, category, start_letters='ABC'):
        for start_letter in start_letters:
            print(self.sample(category, start_letter))
