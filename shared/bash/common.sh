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
