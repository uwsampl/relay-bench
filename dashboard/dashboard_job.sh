#!/bin/bash
#
# Builds TVM from scratch, runs the oopsla benchmarks, makes graphs,
# stores the data and graphs in /var/tmp, and creates a dashboard webpage
webhook_url=$1
ping_users=$2

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
python3 analyze.py --data-dir "${bundle_dir_path}/raw_data" --output-dir "${bundle_dir_path}"

# we will keep analyzed data over time
cp "${bundle_dir_path}/data.json" "${share_store_path}/data.json"
cp "${share_store_path}/data.json" "${share_store_path}/analyzed_data/data_${datestr}.json"
python3 visualize.py --data-dir "${share_store_path}/analyzed_data" --output-dir "${share_store_path}/graph"

# build bundle directory structure and fill it with data
cd "$script_dir"
cp jerry.jpg "${share_store_path}"

# generate static website in bundle to view its data
python3 gen_webpage.py --graph-dir "$share_store_path/graph" --out-dir "$share_store_path"

# post to slack
python3 slack_integration.py --data-dir "${share_store_path}" --post-webhook "${webhook_url}" --ping-users "${ping_users}"
