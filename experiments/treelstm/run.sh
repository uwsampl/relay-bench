#!/bin/bash
config_dir=$1
data_dir=$2

source $BENCHMARK_DEPS/bash/common.sh

include_shared_python_deps
add_to_pythonpath $(pwd)

function check_data_exists {
    declare -a dirs=("checkpoints"
                     "data"
                     "data/sick/dev"
                     "data/sick/test"
                     "data/sick/train")

    declare -a files=("data/glove/glove.840B.300d.pth"
                      "data/glove/glove.840B.300d.txt"
                      "data/glove/glove.840B.300d.vocab"
                      "data/sick/sick_dev.pth"
                      "data/sick/sick_embed.pth"
                      "data/sick/sick_test.pth"
                      "data/sick/sick_train.pth"
                      "data/sick/SICK_test_annotated.txt"
                      "data/sick/SICK_train.txt"
                      "data/sick/SICK_trial.txt"
                      "data/sick/sick.vocab"
                      "data/sick/vocab-cased.txt"
                      "data/sick/vocab.txt")

    declare -a data_files=("a.cparents"
                           "a.parents"
                           "a.rels"
                           "a.toks"
                           "a.txt"
                           "b.cparents"
                           "b.parents"
                           "b.rels"
                           "b.toks"
                           "b.txt"
                           "id.txt"
                           "sim.txt")

    for dir in "${dirs[@]}"
    do
        if ! [ -d "pt_tlstm/$dir" ]; then
            data_exists=false
            return
        fi
    done

    for file in "${files[@]}"
    do
        if ! [ -f "pt_tlstm/$file" ]; then
            data_exists=false
            return
        fi
    done

    for data_file in "${data_files[@]}"
    do
        if ! [ -f "pt_tlstm/data/sick/dev/$data_file" ] || ! [ -f "pt_tlstm/data/sick/test/$data_file" ] || ! [ -f "pt_tlstm/data/sick/train/$data_file" ]; then
            data_exists=false
            return
        fi
    done

    data_exists=true
    return
}


# fetch data and preprocess if it is not there
# note: have to run the script from its own directory because it uses
# relative addresses (bad)
check_data_exists
if ! $data_exists; then
    cd pt_tlstm
    wrap_script_status $data_dir fetch_and_preprocess.sh
    cd ..
fi

python_run_trial "run_pt.py" $config_dir $data_dir

# Because the AoT compiler spawns a lot of subprocesses and potentially
# leaks memory, we're going to spawn each dataset's run as a separate
# process to minimize the chance of running out of memory. Very ugly.
declare -a datasets=("dev"
                     "test"
                     "train")
for dataset in "${datasets[@]}"
do
    python_run_trial "run_relay.py" $config_dir $data_dir "--dataset" "$dataset"
done
