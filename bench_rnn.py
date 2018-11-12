import cProfile
import time
import math
from rnn import pytorch, language_data as data, relay

N_HIDDEN = 128

def time_since(since):
    now = time.time()
    ms = round(1000 * (now - since))
    s = math.floor(ms / 1000)
    m = math.floor(s / 60)
    return '%dm %ds %dms' % (m, s % 60, ms % 1000)

def bench_forward(input_size, hidden_size, output_size):
    # Relay
    relay_rnn = relay.char_rnn_generator.RNNCellOnly(data.N_LETTERS, hidden_size, data.N_LETTERS)
    relay_start = time.time()
    relay.samples(relay_rnn, 'Russian', 'RUS')
    relay.samples(relay_rnn, 'German', 'GER')
    relay.samples(relay_rnn, 'Spanish', 'SPA')
    relay.samples(relay_rnn, 'Chinese', 'CHI')
    print("time of relay: " + time_since(relay_start))

    # relay_loop_start = time.time()
    # relay_rnn = relay.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
    # relay_rnn.samples('Russian', 'RUS')
    # relay_rnn.samples('German', 'GER')
    # relay_rnn.samples('Spanish', 'SPA')
    # relay_rnn.samples('Chinese', 'CHI')
    # print("time of relay: " + time_since(relay_loop_start))

    # PyTorch
    pytorch_rnn = pytorch.char_rnn_generator.RNN(input_size, hidden_size, output_size)
    pytorch_start = time.time()
    pytorch.samples(pytorch_rnn, 'Russian', 'RUS')
    pytorch.samples(pytorch_rnn, 'German', 'GER')
    pytorch.samples(pytorch_rnn, 'Spanish', 'SPA')
    pytorch.samples(pytorch_rnn, 'Chinese', 'CHI')
    print("time of PyTorch: " + time_since(pytorch_start))

def main():
    bench_forward(data.N_LETTERS, N_HIDDEN, data.N_LETTERS)

if __name__ == "__main__":
    main()
