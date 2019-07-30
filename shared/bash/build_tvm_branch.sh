#!/bin/bash
# Script for checking out the specified TVM branch
# First argument should be either "origin" or a remote URL
# Second should be the branch name
remote="$1"
branch="$2"

cd "$TVM_HOME"

remote_name="origin"
if [[ $remote != "origin" ]]; then
    remote_name="other_repo"
    git remote add "$remote_name" "$remote"
    git fetch "$remote_name" "$branch"
fi

git checkout "$remote_name/$branch"
make -j

if [[ $remote != "origin" ]]; then
    git remote rm "$remote_name"
fi
