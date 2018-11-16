import cProfile
import time
import math
from rnn import pytorch, language_data as data, relay
from benchmark import avg_time_since

def bench_forward(hidden_size, iterations=1000):
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

    relay_rnn = relay.char_rnn_generator.RNNLoop(data.N_LETTERS, hidden_size, data.N_LETTERS)
    relay_loop_start = time.time()
    for i in range(iterations):
        relay_rnn.samples('Russian', 'RUS')
        relay_rnn.samples('German', 'GER')
        relay_rnn.samples('Spanish', 'SPA')
        relay_rnn.samples('Chinese', 'CHI')
    print("average iteration time of relay loop: " + avg_time_since(relay_loop_start, iterations))

    pytorch_start = time.time()
    for i in range(iterations):
        # PyTorch
        pytorch_rnn = pytorch.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
        pytorch.samples(pytorch_rnn, 'Russian', 'RUS')
        pytorch.samples(pytorch_rnn, 'German', 'GER')
        pytorch.samples(pytorch_rnn, 'Spanish', 'SPA')
        pytorch.samples(pytorch_rnn, 'Chinese', 'CHI')
    print("average time of PyTorch: " + avg_time_since(pytorch_start, iterations))

def main():
    # cProfile.run('bench_forward(data.N_LETTERS, N_HIDDEN, data.N_LETTERS, 50)')
    for N_HIDDEN in [16, 32, 64, 128, 256, 512]:
        bench_forward(N_HIDDEN, 100)

if __name__ == "__main__":
    main()
