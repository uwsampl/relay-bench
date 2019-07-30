#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

python_run_trial "run_keras.py" $config_dir $data_dir

# This benchmark requires a specific branch of Beacon
rm -rf ./beacon
git clone git@github.com:MarisaKirisame/beacon.git
add_to_pythonpath $(pwd)/beacon

python_run_trial "run_relay.py" $config_dir $data_dir
