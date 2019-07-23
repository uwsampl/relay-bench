#!/bin/bash
#
# Builds dashboard deps, runs the benchmark infrastructure,
# builds the webpage, and invokes the slack integration
dashboard_home=$1

# store path to this script
cd "$(dirname "$0")"
script_dir=$(pwd)

export TVM_HOME=~/dashboard-tvm
# ensure the newly-pulled tvm will be on the Python path
export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}"

# build a fresh TVM from scratch
rm -rf "$TVM_HOME"
git clone --recursive https://github.com/dmlc/tvm "$TVM_HOME"
mkdir "$TVM_HOME/build"
cp config.cmake "$TVM_HOME/build"
cd "$TVM_HOME"
make -j 32

aot_path=~/dashboard-aot
# ensure relay AOT will be on the Python path
export PYTHONPATH="$aot_path:${PYTHONPATH}"

# pull in a new relay AOT compiler
rm -rf "$aot_path"
git clone https://github.com/uwsampl/relay-aot.git "$aot_path"

# ensure CUDA will be present in the path
export PATH="/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}"

# need CUDA LD libraries too
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

cd $script_dir/..
experiments_dir=$(pwd)/experiments

# export because benchmarks may need it
export BENCHMARK_DEPS=$(pwd)/shared
# allow using the shared Python libraries for the dashboard infra
source $BENCHMARK_DEPS/bash/common.sh
include_shared_python_deps

cd $script_dir
python3 dashboard.py --home-dir $dashboard_home --experiments-dir $experiments_dir
python3 gen_webpage.py --dash-home-dir "$dashboard_home" --graph-dir "$dashboard_home/graph" --out-dir "$dashboard_home/website"
python3 slack_integration.py --home-dir $dashboard_home
