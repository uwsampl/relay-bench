#!/bin/bash
data_dir=$1
dest_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
#add_to_pythonpath $(pwd)

python3 visualize.py --data-dir $data_dir --output-dir $dest_dir
