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
    def comput(self):
        pass

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
#         i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + self.b_i
#         f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + self.b_f
#         o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + self.b_o
#         # activations
#         i_t.sigmoid_()
#         f_t.sigmoid_()
#         o_t.sigmoid_()
#         # cell computations
#         c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
#         c_t.tanh_()
#         c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
#         h_t = th.mul(o_t, th.tanh(c_t))
#         # Reshape for compatibility
#         h_t = h_t.view(h_t.size(0), 1, -1)
#         c_t = c_t.view(c_t.size(0), 1, -1)
#         return h_t, (h_t, c_t)

def main():
    slstm = SlowLSTM()
    pass

if __name__ == "__main__":
    main()
