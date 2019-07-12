#!/bin/bash
config_dir=$1
data_dir=$2

export PYTHONPATH=$BENCHMARK_DEPS/python:${PYTHONPATH}
python3 relay_opt.py --config-dir $config_dir --output-dir $data_dir
