# All-in-one Docker Container setup

By building this docker image and running the container, you will have:
- [TVM](https://tvm.ai) with llvm support
- [MLPerf](http://www.mlperf.org)'s inference part
- Relay-bench dashboard

it is recommended to mount ML model dataset instead of downloading/building inside container.

### TVM environment

Set following environment variable before you running TVM Python program.

```
$ export TVM_HOME=/opt/tvm
$ export PYTHONPATH="$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}"
```
