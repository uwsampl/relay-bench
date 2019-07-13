#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

script_dir "$0"
export PYTHONPATH=$BENCHMARK_DEPS/python:${PYTHONPATH}
export PYTHONPATH=$dir_val:${PYTHONPATH}

python3 $dir_val/relay_opt.py --config-dir $config_dir --output-dir $data_dir
