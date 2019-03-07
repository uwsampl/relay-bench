
# Benchmark script for tensorflow lite

## Tensorflow version
version: 1.9
commit: fe7d1d9447a31562acb

## Convert Model
On host.
```bash
python3 convert.py
```
This script will copy \*.tflite model files to devices.

## Build 
on device

### Build tensorflow lite
* for rpi3b
```bash
bash download_dependencies.sh
bash build_rpi_lib.sh 
```

* for aarch64 (hikey960, rk3399)
in 'rpi\_makefile.inc': armv7 -> armv8, remove `-mfpu=...`
```bash
bash download_dependencies.sh
bash build_rpi_lib.sh 
```
### Build benchmark program
```bash
cd ~/autotvm-exp/baseline/tflite
cp label_image.cc ~/tensorflow/tensorflow/contrib/lite/examples/label_image/
cd ~/tensorflow/tensorflow/contrib/lite/examples/label_image/
g++ -std=c++11 -O3 label_image.cc bitmap_helpers.cc -o label_image -I../../../../../ -I../../downloads/flatbuffers/include -ldl -lpthread ../../gen/lib/rpi_armv7/libtensorflow-lite.a
cp label_image ~/autotvm-exp/baseline/tflite
cd ~/autotvm-exp/baseline/tflite
```

## Benchmark
```bash
python bench.py --target rpi3b-cpu
```

