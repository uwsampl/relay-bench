#!/bin/bash
#
# Builds dashboard deps, runs the benchmark infrastructure,
# builds the webpage, and invokes the slack integration
#
# Arguments (must be in this order):
# dashboard home (mandatory)
# rebuild dashboard tvm (optional, true by default)
# experiment dir (optional, assumed to experiments in this repo)
# subsystem dir (optional, assumed to subsystem in this repo)
dashboard_home=$1

# store path to this script
cd "$(dirname "$0")"
script_dir=$(pwd)
experiments_dir=$script_dir/../experiments
subsystem_dir=$script_dir/../subsystem
rebuild_dashboard_tvm=true
if [ "$#" -ge 2 ]; then
    rebuild_dashboard_tvm="$2"
    if [ "$rebuild_dashboard_tvm" != true ]; then
        rebuild_dashboard_tvm=false
    fi
fi
if [ "$#" -ge 3 ]; then
   experiments_dir="$3"
fi
if [ "$#" -ge 4 ]; then
   subsystem_dir ="$4"
fi


export TVM_HOME=~/dashboard-tvm
# ensure the newly-pulled tvm will be on the Python path
export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}"

# build a fresh TVM from scratch
if [ $rebuild_dashboard_tvm = true ]; then
    rm -rf "$TVM_HOME"
    git clone --recursive https://github.com/dmlc/tvm "$TVM_HOME"
    mkdir "$TVM_HOME/build"
    cp config.cmake "$TVM_HOME/build"
    cd "$TVM_HOME"
    make -j 32
fi

aot_path=~/dashboard-aot
# ensure relay AOT will be on the Python path
export PYTHONPATH="$aot_path:${PYTHONPATH}"

# pull in a new relay AOT compiler
rm -rf "$aot_path"
git clone https://github.com/uwsampl/relay-aot.git "$aot_path"

# need /usr/local/bin in PATH
export PATH="/usr/local/bin${PATH:+:${PATH}}"

# ensure CUDA will be present in the path
export PATH="/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}"

# need CUDA LD libraries too
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

cd $script_dir/..

# export because benchmarks may need it
export BENCHMARK_DEPS=$(pwd)/shared
# allow using the shared Python libraries for the dashboard infra
source $BENCHMARK_DEPS/bash/common.sh
include_shared_python_deps

cd $script_dir
python3 dashboard.py --home-dir "$dashboard_home" --experiments-dir "$experiments_dir" --subsystem-dir "$subsystem_dir"
