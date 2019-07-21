#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

export PYTHONPATH="$TVM_HOME/vta/python:$PYTHONPATH"

python3 run.py --config-dir $config_dir --output-dir $data_dir
