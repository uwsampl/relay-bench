#!/bin/bash
config_dir=$1
setup_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

# ensure the setup scripts are in a fresh state
rm -rf pt_tlstm/checkpoints
rm -rf pt_tlstm/data
rm -rf pt_tlstm/lib/stanford*
rm -rf pt_tlstm/lib/*.class

cp -r pt_tlstm "$setup_dir"
cd "$setup_dir/pt_tlstm"
wrap_script_status "$setup_dir" "fetch_and_preprocess.sh"
emit_status_file true "success" "$setup_dir"
