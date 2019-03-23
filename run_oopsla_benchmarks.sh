#!/bin/bash

# have to run TF by itself first because it hogs all the GPU memory
# otherwise
python3 oopsla_benchmarks/cnn_trials.py --n-times-per-input 1000 --skip-pytorch --skip-relay --skip-nnvm
# no run everything besides TF
python3 oopsla_benchmarks/cnn_trials.py --n-times-per-input 1000 --skip-tf

# RNN trials: Relay AOT unfortunately causes a crash if it is
# run more than once in the same Python process so all runs of AOT
# must be done separately
# Only running on CPU because GPU has not been implemented for Pytorch example
python3 oopsla_benchmarks/rnn_trials --n-times-per-input 100 --no-gpu --no-aot
python3 oopsla_benchmarks/rnn_trials --n-times-per-input 100 --no-gpu --no-intp --skip-pytorch --no-cell --append-relay-data
python3 oopsla_benchmarks/rnn_trials --n-times-per-input 100 --no-gpu --no-intp --skip-pytorch --no-loop --append-relay-data
