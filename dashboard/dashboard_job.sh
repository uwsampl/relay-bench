#!/bin/bash
#
# Builds TVM from scratch, runs the oopsla benchmarks, makes graphs,
# stores the data and graphs in /var/tmp, and creates a dashboard webpage

# store path to this script
cd "$(dirname "$0")"
script_dir=$(pwd)

# build TVM
rm -rf ~/dashboard-tvm
git clone --recursive https://github.com/dmlc/tvm ~/dashboard-tvm
mkdir ~/dashboard-tvm/build
cp dashboard/config.cmake ~/dashboard-tvm/build
cd ~/dashboard-tvm; make -j 32

# have to set the newly-pulled tvm to be the one called from python
export TVM_HOME=~/dashboard-tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}

# make a timestamped directory to copy all data and graphs over to
datestr="$(date +"%m-%d-%Y-%H%M")"
bundle_dir_path="/var/tmp/benchmarks_$datestr"
mkdir "$bundle_dir_path"
echo "storing bundle in \"$bundle_dir_path\""

# move to parent directory of this script
cd "$script_dir"/..
./run_oopsla_benchmarks.sh
python3 visualize.py

# TODO: parameterize the eval scripts so we can direct output to a custom dir,
# rather than moving the files in this script.

# build bundle directory structure and fill it with data
cd "$script_dir"
mkdir -p "${bundle_dir_path}/raw_data"
cp ../*.csv "${bundle_dir_path}/raw_data"
mkdir -p "${bundle_dir_path}/graph"
cp ../*.png "${bundle_dir_path}/graph"
cp jerry.jpg "${bundle_dir_path}"

# generate static website in bundle to view its data
python3 gen_webpage.py "--out-dir=${bundle_dir_path}"
