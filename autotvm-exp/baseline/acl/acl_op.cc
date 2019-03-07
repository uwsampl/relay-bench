/*
 * Get gflops of single op.
 *
 * Build: g++ mali_acl_op.cc -std=c++11 -I. -Iinclude -Lbuild -Lbuild/opencl-1.2-stubs -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <unistd.h>

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;


struct GemmWorkload {
    std::string wkl_name;
    size_t n;
    size_t l;
    size_t m;
    std::string dtype;
};

struct Conv2dWorkload {
    std::string wkl_name;
    size_t batch_size;
    size_t height;
    size_t in_filter;
    size_t out_filter;
    size_t hkernel;
    size_t hstride;
    size_t hpad;
    std::string dtype;
};

struct DepthwiseConvWorkload {
    std::string wkl_name;
    size_t n;
    size_t height;
    size_t in_filter;
    int channel_m;
    size_t hkernel;
    size_t hpad;
    size_t hstride;
    std::string dtype;
};



// write result to uniform baseline log file
void log_value(std::string target, std::string device, std::string task_name, std::string method,
               std::string value, std::string outfile="tmp.log") {
    std::stringstream ss;

    ss << target << "\t"
       << device << "\t"
       << task_name << "\t"
       << method << "\t"
       << value << "\t"
       << 0;

    std::ofstream fout(outfile, std::ios::app);
    fout << ss.str() << std::endl;
    std::cout << ss.str() << std::endl;
}

// convert std::vector to python's list in string format
template <typename T>
std::string to_python_list(std::vector<T> array) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < array.size(); i++) {
        if (i != 0)
            ss << ", ";
        ss << array[i];
    }
    ss << "]";
    return ss.str();
}

// transform dtype to format in arm compute
Format DtypeToFormat(std::string dtype) {
    if (dtype == "float" || dtype == "float32")
        return Format::F32;
    else if (dtype == "float16")
        return Format::F16;
    else {
        std::cerr << "Unsupported type: " << dtype << std::endl;
        exit(-1);
    }
}


// measure the cost and gflops of gemm
std::pair<double, double> MeasureGemm(GemmWorkload w, int times) {
    Format format = DtypeToFormat(w.dtype);
    int n = w.n, l = w.l, m = w.m;

    CLTensor a, b, dst;

    // init OpenCL
    CLScheduler::get().default_init();

    // allocate tensors
    a.allocator()->init(TensorInfo(l, n, format));
    b.allocator()->init(TensorInfo(m, l, format));
    dst.allocator()->init(TensorInfo(m, n, format));
    a.allocator()->allocate();
    b.allocator()->allocate();
    dst.allocator()->allocate();

    // configure gemm function
    CLGEMM gemm;
    gemm.configure(&a, &b, nullptr, &dst, 1.0, 0.0);

    // run test
    gemm.run(); gemm.run();

    CLScheduler::get().sync();
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
        gemm.run();
    }
    CLScheduler::get().sync();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // calcuate gflops
    std::chrono::duration<double> fp_ms = end - begin;
    double cost = fp_ms.count() / times;
    return std::make_pair(cost, 2.0 * n * l * m / (1e9) / cost);
}

// measure the cost and gflops of 2d convolution
std::pair<double, double> MeasureConv2d(const Conv2dWorkload &w, int times) {
    Format format = DtypeToFormat(w.dtype);

    CLTensor input, weight, output;
    PadStrideInfo conv_info(w.hstride, w.hstride, w.hpad, w.hpad);

    // init OpenCL
    CLScheduler::get().default_init();

    // allocate tensors
    input.allocator()->init(TensorInfo(TensorShape(w.height, w.height, w.in_filter), format));
    weight.allocator()->init(TensorInfo(TensorShape(w.hkernel, w.hkernel, w.in_filter, w.out_filter), format));
    size_t h_out = (w.height - w.hkernel + w.hpad * 2) / w.hstride + 1;
    size_t w_out = h_out;
    output.allocator()->init(TensorInfo(TensorShape(w_out, h_out, w.out_filter), format));
    input.allocator()->allocate();
    weight.allocator()->allocate();
    output.allocator()->allocate();

    // configure conv function
    CLConvolutionLayer conv2d;
    conv2d.configure(&input, &weight, nullptr, &output, conv_info);

    // run test
    conv2d.run(); conv2d.run(); conv2d.run(); conv2d.run();

    CLScheduler::get().sync();
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
        conv2d.run();
    }
    CLScheduler::get().sync();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // calcuate gflops
    std::chrono::duration<double> fp_ms = end - begin;
    double cost = fp_ms.count() / times;
    return std::make_pair(cost, 2.0 * w_out * h_out * w.out_filter *
                          w.hkernel * w.hkernel * w.in_filter / 1e9 / cost);
}

// measure the cost and gflops of depthwise convolution
std::pair<double, double> MeasureDepthwiseConv(const DepthwiseConvWorkload &w, int times) {
    Format format = DtypeToFormat(w.dtype);

    CLTensor input, weight, output;
    PadStrideInfo conv_info(w.hstride, w.hstride, w.hpad, w.hpad);

    // init OpenCL
    CLScheduler::get().default_init();

    // allocate tensors
    input.allocator()->init(TensorInfo(TensorShape(w.height, w.height, w.in_filter), format));
    weight.allocator()->init(TensorInfo(TensorShape(w.hkernel, w.hkernel, w.in_filter), format));
    size_t h_out = (w.height - w.hkernel + w.hpad * 2) / w.hstride + 1;
    output.allocator()->init(TensorInfo(TensorShape(h_out, h_out, w.in_filter), format));
    input.allocator()->allocate();
    weight.allocator()->allocate();
    output.allocator()->allocate();

    // configure function
    CLDepthwiseConvolutionLayer conv2d;
    conv2d.configure(&input, &weight, nullptr, &output, conv_info);

    // run test
    conv2d.run(); conv2d.run(); conv2d.run(); conv2d.run();

    CLScheduler::get().sync();
    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < times; i++) {
        conv2d.run();
    }
    CLScheduler::get().sync();
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

    // calcuate gflops
    std::chrono::duration<double> fp_ms = end - begin;
    double cost = fp_ms.count() / times;
    return std::make_pair(cost, 2.0 * h_out * h_out *
                          w.hkernel * w.hkernel * w.in_filter / 1e9 / cost);
}



int main(int argc, const char **argv)
{
    GemmWorkload gemm_wkls[] = {
        {"gemm-1024", 1024, 1024, 1024, "float32"},
    };


    Conv2dWorkload conv2d_wkls[] = {
        {"resnet.C1.B1",  1, 224, 3,   64,  7, 2, 3, "float32"},
        {"resnet.C2.B1",  1, 56,  64,  64,  3, 1, 1, "float32"},
        {"resnet.C3.B1",  1, 56,  64,  64,  1, 1, 0, "float32"},
        {"resnet.C4.B1",  1, 56,  64,  128, 3, 2, 1, "float32"},
        {"resnet.C5.B1",  1, 56,  64,  128, 1, 2, 0, "float32"},
        {"resnet.C6.B1",  1, 28,  128, 128, 3, 1, 1, "float32"},
        {"resnet.C7.B1",  1, 28,  128, 256, 3, 2, 1, "float32"},
        {"resnet.C8.B1",  1, 28,  128, 256, 1, 2, 0, "float32"},
        {"resnet.C9.B1",  1, 14,  256, 256, 3, 1, 1, "float32"},
        {"resnet.C10.B1", 1, 14,  256, 512, 3, 2, 1, "float32"},
        {"resnet.C11.B1", 1, 14,  256, 512, 1, 2, 0, "float32"},
        {"resnet.C12.B1", 1, 7,   512, 512, 3, 1, 1, "float32"},
    };

    DepthwiseConvWorkload dc_wkls[] = {
        {"mobilenet.D1.B1", 1, 112, 32, 1, 3, 1, 1, "float32"},
        {"mobilenet.D2.B1", 1, 112, 64, 1, 3, 1, 2, "float32"},
        {"mobilenet.D3.B1", 1, 56, 128, 1, 3, 1, 1, "float32"},
        {"mobilenet.D4.B1", 1, 56, 128, 1, 3, 1, 2, "float32"},
        {"mobilenet.D5.B1", 1, 28, 256, 1, 3, 1, 1, "float32"},
        {"mobilenet.D6.B1", 1, 28, 256, 1, 3, 1, 2, "float32"},
        {"mobilenet.D7.B1", 1, 14, 512, 1, 3, 1, 1, "float32"},
        {"mobilenet.D8.B1", 1, 14, 512, 1, 3, 1, 2, "float32"},
        {"mobilenet.D9.B1", 1, 7, 1024, 1, 3, 1, 1, "float32"},
    };

    int n_ave_curve = 5;
    int sleep_ct = 3;

    double cost, gflops;

    for (size_t i = 0; i < sizeof(gemm_wkls) / sizeof(gemm_wkls[0]); i++) {
        GemmWorkload &w = gemm_wkls[i];
        std::vector<double> costs;

        for (size_t t = 0; t < n_ave_curve; t++) {
            std::tie(cost, gflops) = MeasureGemm(w, 10);
            sleep(sleep_ct);
            costs.push_back(cost);
        }

        std::cout << cost << std::endl;
        log_value("opencl", "Mali-T860", w.wkl_name, "ARMComputeLib-" + w.dtype,
                  to_python_list(costs));
    }

    for (size_t i = 0; i < sizeof(conv2d_wkls) / sizeof(conv2d_wkls[0]); i++) {
        Conv2dWorkload &w = conv2d_wkls[i];
        std::vector<double> costs;

        for (size_t t = 0; t < n_ave_curve; t++) {
            std::tie(cost, gflops) = MeasureConv2d(w, 400);
            sleep(sleep_ct);
            costs.push_back(cost);
        }

        std::cout << cost << std::endl;
        log_value("opencl", "Mali-T860", w.wkl_name, "ARMComputeLib-" + w.dtype,
                  to_python_list(costs));
    }

    for (size_t i = 0; i < sizeof(dc_wkls) / sizeof(dc_wkls[0]); i++) {
        DepthwiseConvWorkload &w = dc_wkls[i];
        std::vector<double> costs;

        for (size_t t = 0; t < n_ave_curve; t++) {
            std::tie(cost, gflops) = MeasureDepthwiseConv(w, 200);
            sleep(sleep_ct);
            costs.push_back(cost);
        }
	
	std::cout << cost << std::endl;
        log_value("opencl", "Mali-T860", w.wkl_name, "ARMComputeLib-" + w.dtype,
                  to_python_list(costs));
    }

    return 0;
}

