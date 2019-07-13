#!/bin/bash
#
# Builds dashboard deps, runs the benchmark infrastructure,
# builds the webpage, and invokes the slack integration

# store path to this script
cd "$(dirname "$0")"
script_dir=$(pwd)

# build TVM
rm -rf ~/dashboard-tvm
git clone --recursive https://github.com/dmlc/tvm ~/dashboard-tvm
mkdir ~/dashboard-tvm/build
cp config.cmake ~/dashboard-tvm/build
cd ~/dashboard-tvm; make -j 32

# pull in a new relay AOT compiler
rm -rf ~/dashboard-aot
git clone https://github.com/uwsampl/relay-aot.git ~/dashboard-aot

# have to set the newly-pulled tvm to be the one called from python
export TVM_HOME=~/dashboard-tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}

# ensure relay AOT will be on the Python path
aot_path=~/dashboard-aot
export PYTHONPATH=$aot_path:${PYTHONPATH}

# ensure CUDA will be present in the path
export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}

# need CUDA LD libraries too
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

dashboard_home="/share/dashboard"
cd $script_dir/..
experiments_dir=($pwd)/experiments

# export because benchmarks may need it
export BENCHMARK_DEPS=($pwd)/shared

cd $script_dir
python3 dashboard.py --home-dir $dashboard_home --experiments-dir $experiments_dir
python3 gen_webpage.py --home-dir $dashboard_home
python3 slack_integration.py --home-dir $dashboard_home
