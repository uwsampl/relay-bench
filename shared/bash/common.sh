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

