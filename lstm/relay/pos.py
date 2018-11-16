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

class LSTM(Network):
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

def bm():
    lstm = LSTM(16, 32)
    N_ITER = 100
    inp = lstm.prelude.nil()
    for _ in range(N_ITER):
        x = np.random.randn(1, 16).astype('float32')
        inp = (lstm.prelude.cons, x, inp)
    hidden = np.random.randn(1, 32).astype('float32')
    cell = np.random.randn(1, 32).astype('float32')
    t = time.time()
    lstm(inp, hidden, cell)
    print(avg_time_since(t, 1))

if __name__ == '__main__':
    bm()
