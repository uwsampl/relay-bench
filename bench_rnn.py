import cProfile
import time
import math
from rnn import pytorch, language_data as data, relay

N_HIDDEN = 128

def avg_time_since(since, iterations):
    now = time.time()
    ms = round(1000 * ((now - since)/iterations))
    s = math.floor(ms / 1000)
    m = math.floor(s / 60)
    return '%dm %ds %dms' % (m, s % 60, ms % 1000)

def bench_forward(input_size, hidden_size, output_size, iterations=1000):
    relay_rnn = relay.char_rnn_generator.RNNCellOnly(data.N_LETTERS, hidden_size, data.N_LETTERS)
    relay_rnn.warm()
    relay_start = time.time()
    for i in range(iterations):
        # Relay using an RNN Cell
        relay.samples(relay_rnn, 'Russian', 'RUS')
        relay.samples(relay_rnn, 'German', 'GER')
        relay.samples(relay_rnn, 'Spanish', 'SPA')
        relay.samples(relay_rnn, 'Chinese', 'CHI')
    print("average iteration time of relay: " + avg_time_since(relay_start, iterations))

        # relay_loop_start = time.time()
        # relay_rnn = relay.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
        # relay_rnn.samples('Russian', 'RUS')
        # relay_rnn.samples('German', 'GER')
        # relay_rnn.samples('Spanish', 'SPA')
        # relay_rnn.samples('Chinese', 'CHI')
        # print("time of relay: " + time_since(relay_loop_start))

    pytorch_start = time.time()
    for i in range(iterations):
        # PyTorch
        pytorch_rnn = pytorch.char_rnn_generator.RNN(input_size, hidden_size, output_size)
        pytorch.samples(pytorch_rnn, 'Russian', 'RUS')
        pytorch.samples(pytorch_rnn, 'German', 'GER')
        pytorch.samples(pytorch_rnn, 'Spanish', 'SPA')
        pytorch.samples(pytorch_rnn, 'Chinese', 'CHI')
    print("average time of PyTorch: " + avg_time_since(pytorch_start, iterations))

def main():
    # cProfile.run('bench_forward(data.N_LETTERS, N_HIDDEN, data.N_LETTERS, 1000)')
    bench_forward(data.N_LETTERS, N_HIDDEN, data.N_LETTERS, 1)

if __name__ == "__main__":
    main()
