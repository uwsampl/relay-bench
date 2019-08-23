#!/bin/bash
config_dir=$1
setup_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

script_dir "$0"

# initialize language data
cd "$setup_dir"
wrap_command_status "$setup_dir" python3 "$dir_val/language_data.py"
emit_status_file true "success" "$setup_dir"
