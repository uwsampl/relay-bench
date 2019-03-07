
ifeq ($(USE_OPENCL), 1)
	ADD_FLAG = -lOpenCL -DUSE_OPENCL
else
	ADD_FLAG =
endif

all: model

op: acl_op.cc
	g++ acl_op.cc build/utils/*.o -std=c++11 -I. -Iinclude -Lbuild -Lbuild/opencl-1.2-stubs -larm_compute -larm_compute_graph -larm_compute_core -o op -O2 $(ADD_FLAG)

model: acl_model.cc
	g++ acl_model.cc build/utils/*.o -std=c++11 -I. -Iinclude -Lbuild -Lbuild/opencl-1.2-stubs -larm_compute -larm_compute_graph -larm_compute_core -o model -O2 $(ADD_FLAG)

clean:
	rm -rf op model

