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

To run the benchmarks ensure you have the internal `tvm-relay`
repository as well as this repo.

Ensure to first install the Python dependencies using `pipenv install`.

You can then launch a session with the corresponding virtual environment
using `pipenv shell`.

In order to generate the graphs you also need the visualization components,
to do so install `npm install -g vega vega-lite`.

Josh has written a script which will run the evaluation on various
targets and produce data for plotting `python3 tvm_benchmark/generate_graphs.py`.

# Future

Eventually this will contain all experiments for Relay.
