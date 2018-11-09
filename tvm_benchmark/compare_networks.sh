#!/bin/bash 
set -e

python3 tvm_benchmark/x86_cpu_imagenet_bench.py --network $1 --rpc-key foo --repeat 1000 --ir relay --output time

python3 tvm_benchmark/x86_cpu_imagenet_bench.py --network $1 --rpc-key foo --repeat 1000 --ir nnvm --output time
