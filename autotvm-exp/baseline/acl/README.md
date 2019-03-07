# Baseline Scritp for ARM Compute Library

## Build

### end2end for NEON
```bash
make 
```

### end2end for OPENCL
```bash
make USE_OPENCL=1
```

### Single Op for OpenCL
```
make op USE_OPENCL=1
```

## Run
```bash
./model rpi3b-cpu
./model rk3399-cpu
```

