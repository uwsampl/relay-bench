#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

rm -rf smll
rm -rf data
rm -f source.py
git clone git@github.com:uwsampl/smll.git

cd smll
stack run -- compile
cd ..
cp smll/python/source.py ./source.py

python_run_trial "run_pt.py" $config_dir $data_dir
