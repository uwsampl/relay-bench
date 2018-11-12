# relay-bench

A repository containing examples and benchmarks for Relay.

Use [Pipenv](https://github.com/pypa/pipenv) to get setup.

These benchmarks only run on Python 3.X.

## Table of Contents
- `rnn` contains various RNN based benchmarks
   used to compare against PyTorch.
- `tvm_benchmark` contains a copy of TVM's official benchmarks
  ported to Relay, generates comparison numbers for NNVM as well.

# RNN

You can generate data for RNN evaluation by executing
`python3 bench_rnn.py`.

# TVM Benchmark
We can reproduce a set of standard TVM benchmarks used in previous
TVM papers. You can find them in the `tvm_benchmark` directory.

Josh has written a script which will run the evaluation on various
targets and produce data for plotting `python3 tvm_benchmark/generate_graphs.py`.

Eventually this will contain all experiments for Relay.
