from io import open
import glob
import os
import random
import unicodedata
import string
import time
import math
from rnn.language_data import N_CATEGORIES, N_LETTERS, topi_to_letter, letter_to_topi
from rnn.relay.util import categoryTensor, inputTensor
import numpy as np
import tvm
from tvm import relay
from tvm.relay import op
from tvm.relay import create_executor, Module
from tvm.relay.prelude import Prelude
from network import *
import aot

class RNNCellOnly(Network):
    def compute(self, input_size, hidden_size, output_size):
        self.category_var = category = relay.var('category', shape=(1, N_CATEGORIES))
        self.input_var = inp = relay.var('input', shape=(1, input_size))
        self.hidden_var = hidden = relay.var('hidden', shape=(1, hidden_size))
        self.hidden = initialize(self.hidden_var)
        combined = op.concatenate([op.concatenate([category, inp], axis=1), hidden], axis=1)
        hidden = self.linear(N_CATEGORIES + input_size + hidden_size, hidden_size, combined, name='i2h')
        output = self.linear(N_CATEGORIES + input_size + hidden_size, output_size, combined, name='i2o')
        output_combined = op.concatenate([hidden, output], axis=1)
        output = self.linear(hidden_size + output_size, output_size, output_combined, name='o2o')
        #output = op.nn.dropout(output, 0.1) #dropout isnt simplified, commented out for now
        output = op.nn.log_softmax(output, axis=1)
        return [self.category_var, self.input_var, self.hidden_var], relay.Tuple([output, hidden]), None

class RNNLoop(Network):
    def compute(self, input_size, hidden_size, output_size):
        self.category_var = category = relay.var('category', shape=(1, N_CATEGORIES))
        self.inp_topi_var = inp_topi = relay.var('input', shape=(), dtype='int32')
        self.hidden_var = hidden = relay.var('hidden', shape=(1, hidden_size))
        self.hidden = initialize(self.hidden_var)
        n_letter = relay.const(N_LETTERS)
        one_diag = relay.const(np.diag(np.ones(58)).astype('float32'))
        boxed_one = relay.const(np.array([1]).astype('int32'))
        inp = op.take(one_diag, op.multiply(boxed_one, inp_topi), axis=0)
        combined = op.concatenate([op.concatenate([category, inp], axis=1), hidden], axis=1)
        hidden = self.linear(N_CATEGORIES + input_size + hidden_size, hidden_size, combined, name='i2h')
        output = self.linear(N_CATEGORIES + input_size + hidden_size, output_size, combined, name='i2o')
        output_combined = op.concatenate([hidden, output], axis=1)
        output = self.linear(hidden_size + output_size, output_size, output_combined, name='o2o')
        # output = op.nn.dropout(output, 0.1) #attributes has not been registered
        output = op.nn.log_softmax(output, axis=1)
        topi = op.argmax(output)
        body = relay.Tuple([hidden,
                            topi,
                            op.equal(topi, op.subtract(n_letter, relay.const(1)))])
        fwd_para = [self.category_var, self.inp_topi_var, self.hidden_var]
        fwd_func = relay.Function(fwd_para, body)
        self.fwd = relay.Var('fwd')

        max = relay.var('max', shape=(), dtype='int32')
        inp_para = [max] + [copy_var(v) for v in fwd_para]
        fwd_res = self.fwd(*inp_para[1:])
        fwd_res_0 = relay.TupleGetItem(fwd_res, 0)
        fwd_res_1 = relay.TupleGetItem(fwd_res, 1)
        fwd_res_2 = relay.TupleGetItem(fwd_res, 2)
        else_else_branch = self.prelude.cons(fwd_res_1, self.recurse(op.subtract(max, relay.const(1)), inp_para[1], fwd_res_1, fwd_res_0))
        else_branch = relay.If(fwd_res_2, self.prelude.nil(), else_else_branch)
        body = relay.If(op.equal(max, relay.const(0)), self.prelude.nil(), else_branch)
        return inp_para, relay.Let(self.fwd, fwd_func, body), None

    def samples(self, category, start_letters='ABC'):
        for start_letter in start_letters:
            self.sample(category, start_letter)

    def sample(self, category, start_letter='A'):
        category_tensor = categoryTensor(category)
        input = letter_to_topi(start_letter)
        hidden = self.hidden
        output = self(20, category_tensor, input, hidden)
