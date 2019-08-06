#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

# this is to initialize any language data in case it's missing
wrap_command_status "$data_dir" python3 language_data.py

python_run_trial "run_pt.py" $config_dir $data_dir
python_run_trial "run_relay.py" $config_dir $data_dir
