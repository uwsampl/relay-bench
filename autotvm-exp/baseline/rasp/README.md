
tensowflow-lite instructions
============================
tensowflow-lite is part of tensorflow/contrib.

Dedicated instructions for building for rpi-3b are not on the main
tensorflow-lite landing page, but can be found in the github repo
(tensorflow/tensorflow/contrib/lite/g3doc/rpi.md)

It seems that tensorflow-lite is not huge, so building without cross-compilation
is doable (takes a few minutes). This produces the tensorflow-lite.a static
library.

The `label_image` example can be used to do some simple timing experiments.
Note that there does not seem to be a build script supplied for raspberry pi,
but putting the following in a Makefile works:
```
label_image: bitmap_helpers.cc label_image.cc
	g++ -std=c++11 label_image.cc bitmap_helpers.cc -o label_image -I../../../../../ -I../../downloads/flatbuffers/include -ldl -lpthread ../../gen/lib/rpi_armv7/libtensorflow-lite.a
```

To run an end-to-end experiment, prepare a `.bmp` image and load the appropriate
model (`.tflite` file after extracting the archive) from:
`https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md`

CONVERTING TENSORFLOW MODELS TO TFLITE
--------------------------------------
The `mobilenet` and `resnet` scripts each produce a directory containing a
corresponding tensorflow `SavedModel`. These can be converted to the `tflite`
format using the following bazel command format:
`bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- --savedmodel_directory=/path/to/model --output_file=/path/to/output.tflite`
