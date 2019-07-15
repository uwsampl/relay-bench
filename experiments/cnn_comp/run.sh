#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

# first arg: script
# second: config dir
# third: data dir
# Runs the script and exits if it has a nonzero error code;
# this way, each script overwrites the status but the script
# will stop if any fails (and writes a failing status)
function run_framework {
    python3 "$1" --config-dir "$2" --output-dir "$3"
    if [ $? -ne 0 ]; then
        exit 1;
    fi
}

run_framework "run_relay.py" $config_dir $data_dir
run_framework "run_nnvm.py" $config_dir $data_dir
run_framework "run_mxnet.py" $config_dir $data_dir
run_framework "run_tf.py" $config_dir $data_dir
run_framework "run_pt.py" $config_dir $data_dir
