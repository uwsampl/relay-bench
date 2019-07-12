#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

current_dir=$script_dir("$0")
export PYTHONPATH=$BENCHMARK_DEPS/python:${PYTHONPATH}
export PYTHONPATH=$current_dir:${PYTHONPATH}

python3 relay_opt.py --config-dir $config_dir --output-dir $data_dir
