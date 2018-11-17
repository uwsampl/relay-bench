import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.nn import Parameter as P
from torch.autograd import Variable as V
from time import time

class SlowLSTM(nn.Module):

    """
    A pedagogic implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory'
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(SlowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # input to hidden weights
        self.w_xi = P(T(hidden_size, input_size))
        self.w_xf = P(T(hidden_size, input_size))
        self.w_xo = P(T(hidden_size, input_size))
        self.w_xc = P(T(hidden_size, input_size))
        # hidden to hidden weights
        self.w_hi = P(T(hidden_size, hidden_size))
        self.w_hf = P(T(hidden_size, hidden_size))
        self.w_ho = P(T(hidden_size, hidden_size))
        self.w_hc = P(T(hidden_size, hidden_size))
        # bias terms
        self.b_i = T(hidden_size).fill_(0)
        self.b_f = T(hidden_size).fill_(0)
        self.b_o = T(hidden_size).fill_(0)
        self.b_c = T(hidden_size).fill_(0)

        # Wrap biases as parameters if desired, else as variables without gradients
        if bias:
            W = P
        else:
            W = V
        self.b_i = W(self.b_i)
        self.b_f = W(self.b_f)
        self.b_o = W(self.b_o)
        self.b_c = W(self.b_c)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(h.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        i_t = th.mm(x, self.w_xi) + th.mm(h, self.w_hi) + self.b_i
        f_t = th.mm(x, self.w_xf) + th.mm(h, self.w_hf) + self.b_f
        o_t = th.mm(x, self.w_xo) + th.mm(h, self.w_ho) + self.b_o
        # activations
        i_t.sigmoid_()
        f_t.sigmoid_()
        o_t.sigmoid_()
        # cell computations
        c_t = th.mm(x, self.w_xc) + th.mm(h, self.w_hc) + self.b_c
        c_t.tanh_()
        c_t = th.mul(c, f_t) + th.mul(i_t, c_t)
        h_t = th.mul(o_t, th.tanh(c_t))
        # Reshape for compatibility
        h_t = h_t.view(h_t.size(0), 1, -1)
        c_t = c_t.view(c_t.size(0), 1, -1)
        return h_t, (h_t, c_t)

def main():
    N_ITER = 100
    SIZES = [128, 256, 512, 1024, 2048]
    lstms = [
        (SlowLSTM, 'SlowLSTM')
    ]

    for lstm, name in lstms:
        ref_res = []
        cus_res = []
        for size in SIZES:
            x = V(th.rand(1, 1, size))
            hiddens = (V(th.rand(1, 1, size)), V(th.rand(1, 1, size)))
            th.manual_seed(1234)
            ref = nn.LSTM(size, size)
            th.manual_seed(1234)
            cus = lstm(size, size)

            out, h = x, hiddens
            save = []
            ref_start = time()
            for i in range(N_ITER):
                out, h = ref(out, h)
                save.append(out)
            ref_time = time() - ref_start
            ref_res.append(ref_time)

            out, h = x, hiddens
            cus_start = time()
            for i in range(N_ITER):
                out, h = cus(out, h)
            cus_time = time() - cus_start
            cus_res.append(cus_time)

        print('## ', name, ' Benchmark ')
        print(' ')
        print('Inference timings on a single sequence of length', N_ITER, '`.')
        print(' ')
        print('size   | nn.LSTM   | ', name, ' | Speedup')
        print('-------|-----------|-' + '-' * len(name) + '---|--------')
        for size, ref, cus in zip(SIZES, ref_res, cus_res):
            print(size, ('   | %.3f     | %.3f' + ' ' * (len(name) - 4) + '  | %.3f') % (ref, cus, ref / cus))
        print(' ')

if __name__ == '__main__':
    # should not run test_speed, as it is way too similar to char-rnn-generator
    main()
