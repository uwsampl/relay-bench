#!/bin/bash
# Bash utilities for dashboard experiments

# First arg should be the name of the calling script (i.e., $0 from the script)
# Returns the directory of that script
function script_dir() {
    dir=$(pwd)
    cd "$(dirname "$1")"
    ret=$(pwd)
    cd $dir
    return $ret
}
