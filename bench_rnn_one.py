import cProfile
import time
import math
from rnn import pytorch, language_data as data, relay
from benchmark import avg_time_since
import argparse

iterations=1
def bench_forward(hidden_size):
    for aot in [True, False]:
        gpu = False
        aot_str = 'compile' if aot else 'interpret'
        gpu_str = 'gpu' if gpu else 'cpu'
        #relay_rnn = relay.char_rnn_generator.RNNCellOnly(aot, gpu, data.N_LETTERS, hidden_size, data.N_LETTERS)
        for i in range(iterations):
            break
            relay.samples(relay_rnn, 'Russian', 'RUS')
            t = time.time()
            relay.samples(relay_rnn, 'Russian', 'RUS')
            relay.samples(relay_rnn, 'German', 'GER')
            relay.samples(relay_rnn, 'Spanish', 'SPA')
            relay.samples(relay_rnn, 'Chinese', 'CHI')
            flush(f'relay_cell_{aot_str}_{gpu_str}', t)

        relay_rnn = relay.char_rnn_generator.RNNLoop(aot, gpu, data.N_LETTERS, hidden_size, data.N_LETTERS)
        for i in range(iterations):
            relay_rnn.samples('Russian', 'RUS')
            t = time.time()
            relay_rnn.samples('Russian', 'RUS')
            relay_rnn.samples('German', 'GER')
            relay_rnn.samples('Spanish', 'SPA')
            relay_rnn.samples('Chinese', 'CHI')
            flush(f'relay_loop_{aot_str}_{gpu_str}', t)

    pytorch_rnn = pytorch.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
    for i in range(iterations):
        pytorch.samples(pytorch_rnn, 'Russian', 'RUS')
        t = time.time()
        pytorch.samples(pytorch_rnn, 'Russian', 'RUS')
        pytorch.samples(pytorch_rnn, 'German', 'GER')
        pytorch.samples(pytorch_rnn, 'Spanish', 'SPA')
        pytorch.samples(pytorch_rnn, 'Chinese', 'CHI')
        flush('pytorch', t)

parser = argparse.ArgumentParser(description='get hidden')
parser.add_argument('N_HIDDEN', type=int,
                    help='an integer for the accumulator')
parser.add_argument('FILE', type=str, nargs='?', default='rnn-data.csv',
                    help='csv file to append to')
args = parser.parse_args()
f=open(args.FILE, "a")

def flush(name, old_time):
    f.write(f'{name},pipsqueak,{args.N_HIDDEN},{round((time.time() - old_time)*1000)}\n')

bench_forward(args.N_HIDDEN)
