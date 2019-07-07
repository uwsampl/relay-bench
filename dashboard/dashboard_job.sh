#!/bin/bash
#
# Pulls in the most recept relay-bench repo and invokes the
# run_dashboard.sh script to ensure that the latest dashboard will
# always be invoked.

tmp_scripts=~/tmp_dashboard_scripts
rm -rf $tmp_scripts
git clone git@github.com:uwsampl/relay-bench.git $tmp_scripts

# need to set up treelstm data
cd $tmp_scripts
./oopsla_benchmarks/pytorch/rnn/tlstm/fetch_and_preprocess.sh

# run its run script
cd $tmp_scripts/dashboard
./run_dashboard.sh
