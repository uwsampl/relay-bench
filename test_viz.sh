#!/usr/bin/env bash

# store path to this script
cd "$(dirname "$0")"
script_dir=$(pwd)

export BENCHMARK_DEPS=/home/weberlo/relay-bench/shared

#
# NORMAL EXPERIMENTS
#

# All experiments: char_rnn  cnn_comp  gluon_rnns  pass_comparison  relay_opt  relay_to_vta  training_loop  treelstm

#rm -r ~/dashboard-home/graph/char_rnn/*
#cd "$script_dir/experiments/char_rnn"
#./visualize.sh /home/weberlo/dashboard-home/config/char_rnn /share/asplos-data/char_rnn /home/weberlo/dashboard-home/graph/char_rnn/
#
#rm -r ~/dashboard-home/graph/cnn_comp/*
#cd "$script_dir/experiments/cnn_comp"
#./visualize.sh /home/weberlo/dashboard-home/config/cnn_comp /share/asplos-data/cnn_comp /home/weberlo/dashboard-home/graph/cnn_comp/

#rm -r ~/dashboard-home/graph/treelstm/*
#cd "$script_dir/experiments/treelstm"
#./visualize.sh /home/weberlo/dashboard-home/config/treelstm /share/asplos-data/treelstm /home/weberlo/dashboard-home/graph/treelstm

# TODO: Uncomment when this experiment is ready
#exp_name=relay_to_vta
#rm -r "~/dashboard-home/graph/$exp_name/*"
#cd "$script_dir/experiments/$exp_name"
#./visualize.sh "/home/weberlo/dashboard-home/config/$exp_name" "/share/asplos-data/$exp_name" "/home/weberlo/dashboard-home/graph/$exp_name/"

#
# POST-PROCESS EXPERIMENTS
#

#exp_name=opt_comp
#rm -r ~/dashboard-home/graph/$exp_name/*
#cd "$script_dir/post_process/$exp_name"
#./visualize.sh /share/asplos-data "/home/weberlo/dashboard-home/graph/$exp_name"
#
#exp_name=vision_comp
#rm -r ~/dashboard-home/graph/$exp_name/*
#cd "$script_dir/post_process/$exp_name"
#./visualize.sh /share/asplos-data "/home/weberlo/dashboard-home/graph/$exp_name"
#
#exp_name=nlp_comp
#rm -r ~/dashboard-home/graph/$exp_name/*
#cd "$script_dir/post_process/$exp_name"
#./visualize.sh /share/asplos-data "/home/weberlo/dashboard-home/graph/$exp_name"

#exp_name=quant_comp
#rm -r ~/dashboard-home/graph/$exp_name/*
#cd "$script_dir/post_process/$exp_name"
#./visualize.sh /share/asplos-data "/home/weberlo/dashboard-home/graph/$exp_name"

exp_name=pass_comp
rm -r ~/dashboard-home/graph/$exp_name/*
cd "$script_dir/post_process/$exp_name"
./visualize.sh /share/asplos-data "/home/weberlo/dashboard-home/graph/$exp_name"

#exp_name=fpga_comp
#rm -r ~/dashboard-home/graph/$exp_name/*
#cd "$script_dir/post_process/$exp_name"
#./visualize.sh /share/asplos-data "/home/weberlo/dashboard-home/graph/$exp_name"
