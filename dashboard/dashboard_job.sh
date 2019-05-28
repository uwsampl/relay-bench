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
cp config.cmake ~/dashboard-tvm/build
cd ~/dashboard-tvm; make -j 32

# have to set the newly-pulled tvm to be the one called from python
export TVM_HOME=~/dashboard-tvm
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}

# ensure relay AOT will be on the Python path
export PYTHONPATH=/share/relay-aot:${PYTHONPATH}

# ensure CUDA will be present in the path
export PATH=/usr/local/cuda-10.0/bin:/usr/local/cuda-10.0/NsightCompute-1.0${PATH:+:${PATH}}

# need CUDA LD libraries too
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64


# make a timestamped directory to copy all data and graphs over to
datestr="$(date +"%m-%d-%Y-%H%M")"
share_store_path="/share/benchmarks"
bundle_dir_path="/var/tmp/benchmarks_$datestr"
mkdir -p "$share_store_path"
mkdir -p "$bundle_dir_path"
echo "storing bundle in \"$bundle_dir_path\""

# move to parent directory of this script
cd "$script_dir"/..
./run_oopsla_benchmarks.sh "${bundle_dir_path}/raw_data"
python3 visualize.py --data-dir "${bundle_dir_path}/raw_data" --output-dir "${bundle_dir_path}/graph"

# build bundle directory structure and fill it with data
cd "$script_dir"
cp jerry.jpg "${bundle_dir_path}"

# generate static website in bundle to view its data
python3 gen_webpage.py --graph-dir "${bundle_dir_path}/graph" --out-dir "${bundle_dir_path}"
cp -r $bundle_dir_path/* $share_store_path
