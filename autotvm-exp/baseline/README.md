# Baseline system

In baseline system, we use a different log format from the tuning curve format.

## Format

A tsv file, each line is 
```
device, backend, workload_type, workload, method, template, value, time_stamp
```

Example
```
rk3399-cpu, eigen, network, resnet-18.B1, tflite, default, {"costs": [0.123, 0.234]}, 93485.23

1080ti, cuda, network, resnet-18.B1, mxnet, default, {"costs": [0.123, 0.234]}, 93485.23
1080ti, cuda, network, resnet-18.B1, mxnet-trt, default, {"costs": [0.123, 0.234]}, 94385.23

1080ti, cuda, op, resnet-18.C1.B1, xgb-rank, direct, {"costs": [0.0012, 0.0013]}, 34283.234
```

## How to gather baseline

* install dependency on remote machines
* use ssh to update autotvm-exp repo on remote machines
* use ssh to run benchmark scripts
* gather results

