#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

rm -rf pt_tlstm
mv "./setup/pt_tlstm" pt_tlstm
rm -rf setup

python_run_trial "run_pt.py" $config_dir $data_dir

# Because the AoT compiler spawns a lot of subprocesses and potentially
# leaks memory, we're going to spawn each dataset's run as a separate
# process to minimize the chance of running out of memory. Very ugly.
declare -a datasets=("dev"
                     "test"
                     "train")
for dataset in "${datasets[@]}"
do
    # launch interpreter and AoT as separate subprocesses because
    # they seem to leak memory. Also very ugly
    python_run_trial "run_relay.py" $config_dir $data_dir "--dataset" "$dataset" "--method" "intp"
    python_run_trial "run_relay.py" $config_dir $data_dir "--dataset" "$dataset" "--method" "aot"
done
