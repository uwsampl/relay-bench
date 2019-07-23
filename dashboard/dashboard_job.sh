#!/bin/bash
#
# Pulls in the most recept relay-bench repo and invokes the
# run_dashboard.sh script to ensure that the latest dashboard will
# always be invoked.
dashboard_home=/share/dashboard/
tmp_scripts=~/tmp_dashboard_scripts
rm -rf $tmp_scripts
git clone git@github.com:uwsampl/relay-bench.git $tmp_scripts

# run its run script
cd $tmp_scripts/dashboard
./run_dashboard.sh $dashboard_home
