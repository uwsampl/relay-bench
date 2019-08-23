#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

# move language data from the setup directory into here
cp -r ./setup/. .
rm -rf ./setup

python_run_trial "run_pt.py" $config_dir $data_dir
python_run_trial "run_relay.py" $config_dir $data_dir
