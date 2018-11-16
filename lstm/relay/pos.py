import time
from benchmark import avg_time_since
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
from tvm.relay.backend.interpreter import TensorValue
from tvm.relay.prelude import Prelude
from aot import aot
from network import *

class SlowLSTM(Network):
    def compute(self, embedding_dim, hidden_dim):
        fwd_input = relay.var('fwd_input', shape=(1, embedding_dim,))
        fwd_hidden = relay.var('fwd_hidden', shape=(1, hidden_dim,))
        fwd_cell = relay.var('fwd_cell', shape=(1, hidden_dim,))
        self.fwd_cell = relay.Var('fwd_cell')
        combined = op.concatenate2(fwd_input, fwd_hidden, axis=1)
        input_dim = embedding_dim + hidden_dim
        a = op.sigmoid(self.linear(input_dim, hidden_dim, combined, 'a'))
        b = op.sigmoid(self.linear(input_dim, hidden_dim, combined, 'b'))
        c = op.tanh(self.linear(input_dim, hidden_dim, combined, 'c'))
        d = op.sigmoid(self.linear(input_dim, hidden_dim, combined, 'd'))
        new_cell = fwd_cell * a + b * c
        new_hidden = op.tanh(new_cell) * d
        fwd_body = relay.Tuple([new_hidden, new_cell])
        fwd_cell_func = relay.Function([fwd_input, fwd_hidden, fwd_cell], fwd_body)

        self.input = relay.Var('input', self.prelude.l(relay.TensorType(dtype='float32', shape=(1, embedding_dim,))))
        self.hidden = relay.var('hidden', shape=(1, hidden_dim,))
        self.cell = relay.var('cell', shape=(1, hidden_dim,))
        nil_case = relay.Clause(relay.PatternConstructor(self.prelude.nil), self.prelude.nil())
        x = relay.Var('x')
        y = relay.Var('y')
        fwd_res = self.fwd_cell(x, self.hidden, self.cell)
        hidden_output = fwd_res[0]
        cell_output = fwd_res[1]
        cons_case = relay.Clause(relay.PatternConstructor(self.prelude.cons,
                                                          [relay.PatternVar(x), relay.PatternVar(y)]),
                                 self.prelude.cons(hidden_output, self.recurse(y, hidden_output, cell_output)))
        body = relay.Match(self.input, [nil_case, cons_case])
        body = relay.Let(self.fwd_cell, fwd_cell_func, body)
        return [self.input, self.hidden, self.cell], body, None

# class SlowLSTM(nn.Module):

#     """
#     A pedagogic implementation of Hochreiter & Schmidhuber:
#     'Long-Short Term Memory'
#     http://www.bioinf.jku.at/publications/older/2604.pdf
#     """

#     def __init__(self, input_size, hidden_size):
#         super(SlowLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         # input to hidden weights
#         self.w_xi = P(T(hidden_size, input_size))
#         self.w_xf = P(T(hidden_size, input_size))
#         self.w_xo = P(T(hidden_size, input_size))
#         self.w_xc = P(T(hidden_size, input_size))
#         # hidden to hidden weights
#         self.w_hi = P(T(hidden_size, hidden_size))
#         self.w_hf = P(T(hidden_size, hidden_size))
#         self.w_ho = P(T(hidden_size, hidden_size))
#         self.w_hc = P(T(hidden_size, hidden_size))
#         # bias terms
#         self.b_i = T(hidden_size).fill_(0)
#         self.b_f = T(hidden_size).fill_(0)
#         self.b_o = T(hidden_size).fill_(0)
#         self.b_c = T(hidden_size).fill_(0)

#         # Wrap biases as parameters if desired, else as variables without gradients
#         self.b_i = P(self.b_i)
#         self.b_f = P(self.b_f)
#         self.b_o = P(self.b_o)
#         self.b_c = P(self.b_c)
#         self.reset_parameters()

#     def reset_parameters(self):
#         std = 1.0 / math.sqrt(self.hidden_size)
#         for w in self.parameters():
#             w.data.uniform_(-std, std)

#     def forward(self, x, hidden):
#         h, c = hidden
#         h = h.view(h.size(0), -1)
#         c = c.view(h.size(0), -1)
#         x = x.view(x.size(0), -1)
#         # Linear mappings
#         # activations
#         i_t.sigmoid_()
#         f_t.sigmoid_()
#         o_t.sigmoid_()
#         # cell computations
#         c_t.tanh_()
#         c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
#         h_t = th.mul(o_t, th.tanh(c_t))
#         # Reshape for compatibility
#         h_t = h_t.view(h_t.size(0), 1, -1)
#         c_t = c_t.view(c_t.size(0), 1, -1)
#         return h_t, (h_t, c_t)

# class RNNCellOnly(Network):
#     def compute(self, input_size, hidden_size, output_size):
#         self.category_var = category = relay.var('category', shape=(1, N_CATEGORIES))
#         self.input_var = inp = relay.var('input', shape=(1, input_size))
#         self.hidden_var = hidden = relay.var('hidden', shape=(1, hidden_size))
#         self.hidden = initialize(self.hidden_var)
#         combined = op.concatenate2(op.concatenate2(category, inp, axis=1), hidden, axis=1)
#         hidden = self.linear(N_CATEGORIES + input_size + hidden_size, hidden_size, combined, name='i2h')
#         output = self.linear(N_CATEGORIES + input_size + hidden_size, output_size, combined, name='i2o')
#         output_combined = op.concatenate2(hidden, output, axis=1)
#         output = self.linear(hidden_size + output_size, output_size, output_combined, name='o2o')
#         #output = op.nn.dropout(output, 0.1) #dropout isnt simplified, commented out for now
#         output = op.nn.log_softmax(output, axis=1)
#         return [self.category_var, self.input_var, self.hidden_var], relay.Tuple([output, hidden]), None

#     def warm(self):
#         self(initialize(self.category_var), initialize(self.input_var), initialize(self.hidden_var))

# class RNNLoop(Network):
#     def compute(self, input_size, hidden_size, output_size):
#         self.category_var = category = relay.var('category', shape=(1, N_CATEGORIES))
#         self.inp_topi_var = inp_topi = relay.var('input', shape=(), dtype='int32')
#         self.hidden_var = hidden = relay.var('hidden', shape=(1, hidden_size))
#         self.hidden = initialize(self.hidden_var)
#         n_letter = relay.const(N_LETTERS)
#         one_diag = relay.const(np.diag(np.ones(58)).astype('float32'))
#         boxed_one = relay.const(np.array([1]).astype('int32'))
#         inp = op.take(one_diag, op.multiply(boxed_one, inp_topi), axis=0)
#         combined = op.concatenate2(op.concatenate2(category, inp, axis=1), hidden, axis=1)
#         hidden = self.linear(N_CATEGORIES + input_size + hidden_size, hidden_size, combined, name='i2h')
#         output = self.linear(N_CATEGORIES + input_size + hidden_size, output_size, combined, name='i2o')
#         output_combined = op.concatenate2(hidden, output, axis=1)
#         output = self.linear(hidden_size + output_size, output_size, output_combined, name='o2o')
#         # output = op.nn.dropout(output, 0.1) #attributes has not been registered
#         output = op.nn.log_softmax(output, axis=1)
#         topi = op.argmax(output)
#         body = relay.Tuple([hidden,
#                             topi,
#                             op.equal(topi, op.subtract(n_letter, relay.const(1)))])
#         fwd_para = [self.category_var, self.inp_topi_var, self.hidden_var]
#         fwd_func = relay.Function(fwd_para, body)
#         self.fwd = relay.Var('fwd')

#         max = relay.var('max', shape=(), dtype='int32')
#         inp_para = [max] + [copy_var(v) for v in fwd_para]
#         fwd_res = self.fwd(*inp_para[1:])
#         fwd_res_1 = fwd_res[1]
#         else_else_branch = self.prelude.cons(fwd_res_1, self.recurse(op.subtract(max, relay.const(1)), inp_para[1], fwd_res_1, fwd_res[0]))
#         else_branch = relay.If(fwd_res[2], self.prelude.nil(), else_else_branch)
#         body = relay.If(op.equal(max, relay.const(0)), self.prelude.nil(), else_branch)
#         return inp_para, relay.Let(self.fwd, fwd_func, body), None

#     def samples(self, category, start_letters='ABC'):
#         for start_letter in start_letters:
#             self.sample(category, start_letter)

#     def sample(self, category, start_letter='A'):
#         category_tensor = categoryTensor(category)
#         input = letter_to_topi(start_letter)
#         hidden = self.hidden
#         output = self(20, category_tensor, input, hidden)

def bm():
    t = time.time()
    slstm = SlowLSTM(16, 32)
    print(avg_time_since(t, 1))

if __name__ == '__main__':
    bm()
