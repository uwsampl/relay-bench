from rnn import pytorch

N_HIDDEN = 128

def bench_forward(input_size, hidden_size, output_size):
    import pdb; pdb.set_trace()
    pytorch.char_rnn_generator.RNN(input_size, hidden_size, output_size)

def main():
    bench_forward(pytorch.char_rnn_generator.N_LETTERS, N_HIDDEN, pytorch.char_rnn_generator.N_LETTERS)

if __name__ == "__main__":
    main()
