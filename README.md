# relay-bench

A repository containing examples and benchmarks for Relay.

Use [Pipenv](https://github.com/pypa/pipenv) to get set up.

These benchmarks only run on Python 3.X.

## Table of Contents
- `oopsla_benchmarks` contains a variety of TVM benchmarks with comparisons to other frameworks.
- `dashboard` contains scripts for running these benchmarks in an overnight benchmark.

# TVM Benchmark
We can reproduce a set of standard TVM benchmarks used in previous
TVM papers. You can find them in the `oopsla_benchmarks` directory.

To run the benchmarks, make sure you have TVM installed (and registered
in the `$TVM_HOME` environmental variable), as well as CUDA 10.0 and
CuDNN 7.5.0 (these dependencies cannot be handled by pipenv).

Be sure to first install the Python dependencies using `pipenv install`.

You can then launch a session with the corresponding virtual environment
using `pipenv shell`. The script `run_oopsla_benchmarks.sh` should take
care of running the benchmarks themselves.

# Future

Eventually this will contain all experiments for Relay.
