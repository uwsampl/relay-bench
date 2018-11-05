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
    relay_start = time.time()
    # Relay
    relay_rnn = relay.char_rnn_generator.RNN(data.N_LETTERS, 128, data.N_LETTERS)
    relay.samples(relay_rnn, 'Russian', 'RUS')
    relay.samples(relay_rnn, 'German', 'GER')
    relay.samples(relay_rnn, 'Spanish', 'SPA')
    relay.samples(relay_rnn, 'Chinese', 'CHI')
    print("time of relay: " + time_since(relay_start))

    # PyTorch
    pytorch_start = time.time()
    pytorch_rnn = pytorch.char_rnn_generator.RNN(input_size, hidden_size, output_size)
    pytorch.samples(pytorch_rnn, 'Russian', 'RUS')
    pytorch.samples(pytorch_rnn, 'German', 'GER')
    pytorch.samples(pytorch_rnn, 'Spanish', 'SPA')
    pytorch.samples(pytorch_rnn, 'Chinese', 'CHI')
    print("time of PyTorch: " + time_since(pytorch_start))

def main():
    bench_forward(data.N_LETTERS, N_HIDDEN, data.N_LETTERS)

if __name__ == "__main__":
    main()
