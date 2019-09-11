#!/bin/bash
config_dir=$1
setup_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

python3 setup.py --config-dir "$config_dir" --setup-dir "$setup_dir"
