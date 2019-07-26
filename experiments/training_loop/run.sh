#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

python_run_trial "run_keras.py" $config_dir $data_dir

# This benchmarks requires a specific branch of Beacon and a specific branch
# of TVM, so we have no choice but to install that Beacon and TVM.
# This is terrible and must be the top priority for changing

# clear pythonpath to prevent bad things from happening
export PYTHONPATH=""
export TVM_HOME=$(pwd)/marisa-tvm
rm -rf $TVM_HOME
git clone --recursive git@github.com:MarisaKirisame/tvm.git $TVM_HOME
mkdir $TVM_HOME/build
cp config.cmake "$TVM_HOME/build"
cd $TVM_HOME
git checkout origin/add-grads
make -j 32
cd ..
export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}"

rm -rf ./beacon
git clone git@github.com:MarisaKirisame/beacon.git
add_to_pythonpath $(pwd)/beacon

include_shared_python_deps
add_to_pythonpath $(pwd)

python_run_trial "run_relay.py" $config_dir $data_dir
