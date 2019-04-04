#!/bin/bash

# have to run TF by itself first because it hogs all the GPU memory
# otherwise
python3 oopsla_benchmarks/cnn_trials.py --n-times-per-input 1000 --skip-pytorch --skip-relay --skip-nnvm --skip-mxnet
# now run everything besides TF
python3 oopsla_benchmarks/cnn_trials.py --n-times-per-input 1000 --skip-tf

# Only running on CPU because GPU has not been implemented for Pytorch example
python3 oopsla_benchmarks/rnn_trials.py --n-times-per-input 100 --no-gpu
