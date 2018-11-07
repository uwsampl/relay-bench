import cProfile
import time
import math
from rnn import pytorch, language_data as data, relay

N_HIDDEN = 1024

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

class SampleRNNBench(Bench):
    def __init__(self, name, sample, extract):
        super().__init__()
        self.name = name
        self.sample = sample
        self.extract = extract

    def prepare(self, samples):
        def compute():
            ret = []
            for sample in samples:
                for l in sample.starting_letters:
                    ret.append(self.sample(sample.category, l))
            def extract():
                ext = []
                for r in ret:
                    ext.append(self.extract(r))
                return ext
            return extract
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
    bench.append(SampleRNNBench("relay", lambda c, s: relay.sample(relay_rnn, c, s), lambda x: x))
    relay_rnn = relay.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
    bench.append(SampleRNNBench("relay with loop", lambda c, s: relay_rnn.sample(c, s), lambda x: x()))
    pytorch_rnn = pytorch.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
    bench.append(SampleRNNBench("pytorch", lambda c, s: pytorch.sample(pytorch_rnn, c, s), lambda x: x))
    for b in bench:
        t, r = b(sample)
        for l in r:
            print(l)
        print("time of " + str(b) + " : " + str(t))

def profile():
    sample = [
        RNNSample('Russian', 'RUS'),
        RNNSample('German', 'GER'),
        RNNSample('Spanish', 'SPA'),
        RNNSample('Chinese', 'CHI')]
    bench = []
    relay_rnn = relay.char_rnn_generator.RNN(data.N_LETTERS, 128, data.N_LETTERS)
    bench.append(SampleRNNBench("relay with loop", lambda c, s: relay_rnn.sample(c, s), lambda x: x()))
    cProfile.runctx('bench[0](sample)', {'bench':bench, 'sample':sample}, {})

def main():
    bench_forward(N_HIDDEN)
    #profile()

if __name__ == "__main__":
    main()
