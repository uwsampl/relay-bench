#!/bin/bash
config_dir=$1
data_dir=$2
dest_dir=$3

source $BENCHMARK_DEPS/bash/common.sh

script_dir "$0"
export PYTHONPATH=$BENCHMARK_DEPS/python:${PYTHONPATH}
export PYTHONPATH=$dir_val:${PYTHONPATH}

python3 $dir_val/relay_analyze.py --config-dir $config_dir --data-dir $data_dir --output-dir $dest_dir
