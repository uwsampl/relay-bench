#!/bin/bash
# Bash utilities for dashboard experiments

# First arg should be the name of the calling script (i.e., $0 from the script)
# Sets dir_val to the directory of the script
function script_dir {
    dir=$(pwd)
    cd "$(dirname "$1")"
    ret=$(pwd)
    cd $dir
    dir_val=$ret
}
export -f script_dir

# Adds the argument to the PYTHONPATH var
function add_to_pythonpath {
    export PYTHONPATH=$1:${PYTHONPATH}
}
export -f add_to_pythonpath

# Adds the shared python deps to the PYTHONPATH var
function include_shared_python_deps {
    add_to_pythonpath "$BENCHMARK_DEPS/python"
}
export -f include_shared_python_deps

# Runs a Python script with the passed arguments and exits if its exit
# code is nonzero
function check_python_exit_code {
    python3 "$@"
    if [ $? -ne 0 ]; then
        exit 1;
    fi
}
export -f check_python_exit_code

# Takes at least three arguments (a Python script meant to run a trial,
# a directory meant to be the config dir, and a directory meant to be
# the output dir) and runs the following:
# python3 script.py --config-dir config_dir --output-dir output_dir [all remaining args]
# Exits if the exit code is nonzero
function python_run_trial {
    check_python_exit_code "$1" "--config-dir" "$2" "--output-dir" "$3" "${@:4}"
}
export -f python_run_trial
