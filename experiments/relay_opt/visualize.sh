#!/bin/bash
config_dir=$1
data_dir=$2
dest_dir=$3

source $BENCHMARK_DEPS/bash/common.sh

current_dir=$script_dir("$0")
export PYTHONPATH=$BENCHMARK_DEPS/python:${PYTHONPATH}
export PYTHONPATH=$current_dir:${PYTHONPATH}

python3 relay_visualize.py --config-dir $config_dir --data-dir $data_dir --output-dir $dest_dir
