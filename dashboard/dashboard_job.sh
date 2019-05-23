#!/bin/bash

# Builds TVM from scratch, runs the oopsla benchmarks, makes graphs,
# stores the data and graphs in /var/tmp, and creates a dashboard webpage

rm -rf ~/dashboard-tvm
cd ..
git clone --recursive https://github.com/dmlc/tvm ~/dashboard-tvm
mkdir ~/dashboard-tvm/build
cp dashboard/config.cmake ~/dashboard-tvm/build
cd ~/dashboard-tvm; make -j 32; cd ../relay-bench

# have to set the newly-pulled tvm to be the one called from python
export TVM_HOME=~/dashboard-tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}

./run_oopsla_benchmarks.sh
python3 visualize.py

# copy all data and graphs over to /var/tmp
datestr=$(date +"%m-%d-%Y-%H%M")
dirname="/var/tmp/benchmarks_$datestr"
mkdir $dirname

cp *.csv $dirname
cp *.png $dirname
