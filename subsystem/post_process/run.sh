#!/bin/bash
config_dir=$1
home_dir=$2
output_dir=$3

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

function run_post_process {
    script="$1"
    check_python_exit_code "$script" --config-dir "$config_dir" --home-dir "$home_dir" --output-dir "$output_dir"
}

run_post_process nlp_comp.py
run_post_process opt_comp.py
run_post_process pass_comp.py
run_post_process vision_comp.py
