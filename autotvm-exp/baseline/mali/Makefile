
all: op model

op: mali_acl_op.cc
	g++ mali_acl_op.cc build/utils/*.o -std=c++11 -I. -Iinclude -Lbuild -Lbuild/opencl-1.2-stubs -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o op

model: mali_acl_model.cc
	g++ mali_acl_model.cc build/utils/*.o -std=c++11 -I. -Iinclude -Lbuild -Lbuild/opencl-1.2-stubs -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL -o model

clean:
	rm op model

