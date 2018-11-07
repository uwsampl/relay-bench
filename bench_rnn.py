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

class Bench:
    """
    A benchmark is a inp -> () -> () -> (time, out).
    The zeroth application take all relevant argument and setup the benchmark (compile, prepare data)
    The first application do all the real work, (which you should time),
    The third application extract the final data.
    """
    def prepare(self, *arg):
        raise NotImplementedError

    def __call__(self, *arg):
        prepared = self.prepare(*arg)
        start = time.time()
        computed = prepared()
        took = time_since(start)
        return (took, computed())

class RNNSample:
    def __init__(self, category, starting_letters):
        self.category = category
        self.starting_letters = starting_letters

class SamplesRNNBench(Bench):
    def __init__(self, name, samples):
        super().__init__()
        self.name = name
        self.samples = samples

    def prepare(self, samples):
        def compute():
            for sample in samples:
                self.samples(sample.category, sample.starting_letters)
            return lambda: None
        return compute

    def __str__(self):
        return self.name

def bench_forward(hidden_size):
    sample = [
        RNNSample('Russian', 'RUS'),
        RNNSample('German', 'GER'),
        RNNSample('Spanish', 'SPA'),
        RNNSample('Chinese', 'CHI')]
    bench = []
    relay_rnn = relay.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
    bench.append(SamplesRNNBench("relay", lambda c, s: relay.samples(relay_rnn, c, s)))
    relay_rnn = relay.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
    bench.append(SamplesRNNBench("relay with loop", lambda c, s: relay_rnn.samples(c, s)))
    pytorch_rnn = pytorch.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
    bench.append(SamplesRNNBench("pytorch", lambda c, s: pytorch.samples(pytorch_rnn, c, s)))
    # Relay
    for b in bench:
        t, r = b(sample)
        print("time of " + str(b) + " : " + str(t))

def main():
    #cProfile.run('bench_forward(data.N_LETTERS, N_HIDDEN, data.N_LETTERS)')
    bench_forward(N_HIDDEN)

if __name__ == "__main__":
    main()
