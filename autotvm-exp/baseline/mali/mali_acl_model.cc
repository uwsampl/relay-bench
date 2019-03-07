/*
 * End-to-End evaluation of models
 *
 * Build: g++ mali_acl_model.cc build/utils/*.o -std=c++11 -I. -Iinclude -Lbuild -Lbuild/opencl-1.2-stubs -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL
 */

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "arm_compute/runtime/Scheduler.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>
#include <unistd.h>

using namespace arm_compute::graph;
using namespace arm_compute::graph_utils;

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


std::unique_ptr<ITensorAccessor> dummy() {
    return arm_compute::support::cpp14::make_unique<DummyAccessor>(1);
}

/*
 * MODEL DEFINITION
 */
void get_vgg16(Graph *graph, arm_compute::DataType type) {
    *graph << Tensor(TensorInfo(TensorShape(224U, 224U, 3U, 1U), 1, type))
          // Layer 1
          << ConvolutionLayer(3U, 3U, 64U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 2
          << ConvolutionLayer(3U, 3U, 64U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 3
          << ConvolutionLayer(3U, 3U, 128U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 4
          << ConvolutionLayer(3U, 3U, 128U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 5
          << ConvolutionLayer(3U, 3U, 256U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 6
          << ConvolutionLayer(3U, 3U, 256U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 7
          << ConvolutionLayer(3U, 3U, 256U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 8
          << ConvolutionLayer(3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 9
          << ConvolutionLayer(3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 10
          << ConvolutionLayer(3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 11
          << ConvolutionLayer(3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 12
          << ConvolutionLayer(3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 13
          << ConvolutionLayer(3U, 3U, 512U, dummy(), dummy(), PadStrideInfo(1, 1, 1, 1))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2, 0, 0)))
          // Layer 14
          << FullyConnectedLayer(4096U, dummy(), dummy())
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 15
          << FullyConnectedLayer(4096U, dummy(), dummy())
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 16
          << FullyConnectedLayer(1000U, dummy(), dummy())
          // Softmax
          << SoftmaxLayer()
          << Tensor(TensorInfo(TensorShape(1000U), 1, type));
}

void get_nature_dqn(Graph *graph, arm_compute::DataType type) {
    *graph << Tensor(TensorInfo(TensorShape(84U, 84U, 4U, 1U), 1, type))
          // Layer 1
          << ConvolutionLayer(8U, 8U, 32U, dummy(), dummy(), PadStrideInfo(4, 4, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 2
          << ConvolutionLayer(4U, 4U, 64U, dummy(), dummy(), PadStrideInfo(2, 2, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 3
          << ConvolutionLayer(3U, 3U, 64U, dummy(), dummy(), PadStrideInfo(1, 1, 0, 0))
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 4
          << FullyConnectedLayer(512U, dummy(), dummy())
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          // Layer 5
          << FullyConnectedLayer(18, dummy(), dummy())
          << Tensor(TensorInfo(TensorShape(18U), 1, type));
}


BranchLayer get_dwsc_node(const std::string &data_path, std::string &&param_path,
                          unsigned int  conv_filt,
                          PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
{
    std::string total_path = "/cnn_data/mobilenet_v1_model/" + param_path + "_";
    SubGraph    sg;
    sg << DepthwiseConvolutionLayer(
                   3U, 3U, dummy(),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   dwc_pad_stride_info,
                   true)
       << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.001f)
       << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
       << ConvolutionLayer( 1U, 1U, conv_filt, dummy(),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), conv_pad_stride_info)
       << BatchNormalizationLayer( dummy(), dummy(), dummy(), dummy(), 0.001f)
       << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));

    return BranchLayer(std::move(sg));
}

void get_mobilenet(Graph *graph, arm_compute::DataType type) {
    std::string data_path; /* Path to the trainable data */

    *graph << Tensor(TensorInfo(TensorShape(224U, 224U, 3U, 1U), 1, type))
          << ConvolutionLayer( 3U, 3U, 32U, dummy(),
              std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
              PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
          << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.001f)
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
          << get_dwsc_node(data_path, "Conv2d_1", 64, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_2", 128, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_3", 128, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_4", 256, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_5", 256, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_6", 512, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_7", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_8", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_9", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_10", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_11", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_12", 1024, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << get_dwsc_node(data_path, "Conv2d_13", 1024, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0))
          << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
          << ConvolutionLayer( 1U, 1U, 1000U, dummy(), dummy(), PadStrideInfo(1, 1, 0, 0))
          << ReshapeLayer(TensorShape(1000U))
          << SoftmaxLayer()
          << Tensor(TensorInfo(TensorShape(1000U), 1, type));
}

void add_residual_unit(Graph *graph, int num_filter, int stride, bool dim_match) {
    SubGraph right;
    right << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.00001f)
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << ConvolutionLayer(3U, 3U, num_filter, dummy(),
                              std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                              PadStrideInfo(stride, stride, 1, 1))
          << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.00001f)
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
          << ConvolutionLayer(3U, 3U, num_filter, dummy(),
                              std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                              PadStrideInfo(1, 1, 1, 1));

    if (dim_match) {
        *graph << ResidualLayer(std::move(right));
    } else {
        SubGraph left;
        left  << ConvolutionLayer(1U, 1U, num_filter, dummy(),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(stride, stride, 1, 1));
        *graph << ResidualLayer(std::move(left), std::move(right));
    }
}


void get_resnet18(Graph *graph, arm_compute::DataType type) {
    int filter_list[] = {64, 64, 128, 256, 512};
    int num_stages = 4;
    int units[] = {2, 2, 2, 2};

    *graph << Tensor(TensorInfo(TensorShape(224U, 224U, 3U, 1U), 1, type))
           << ConvolutionLayer(7U, 7U, filter_list[0], dummy(), dummy(), PadStrideInfo(2, 2, 3, 3))
           << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.00001f)
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
           << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR)));

    for (int i = 0; i < num_stages; i++) {
        int stride = i == 0 ? 1 : 2;
        add_residual_unit(graph, filter_list[i+1], stride, false);
        for (int j = 0; j < units[i] - 1; j++)
            add_residual_unit(graph, filter_list[i+1], 1, true);
    }

    *graph << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.00001f)
           << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
           << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
           << ConvolutionLayer(1U, 1U, 1000U, dummy(), dummy(), PadStrideInfo(1, 1, 0, 0))
           << FlattenLayer()
           << SoftmaxLayer()
           << Tensor(TensorInfo(TensorShape(1000U), 1, type));
}


/*
 * MEASURE FUNCTION
 */

double measure(Graph *graph, int n_times) {
    arm_compute::CLScheduler::get().default_init();
    graph->run(); graph->run();

    arm_compute::CLScheduler::get().sync();
    auto tbegin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_times; i++) {
        graph->run();
    }
    arm_compute::CLScheduler::get().sync();
    auto tend = std::chrono::high_resolution_clock::now();


    double cost = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
    return cost / n_times;
}

double run_case(std::string backend, std::string model, std::string conv_method, std::string dtype) {
    TargetHint            target_hint;
    ConvolutionMethodHint convolution_hint;
    arm_compute::DataType type;

    if (conv_method == "gemm") {
        convolution_hint = ConvolutionMethodHint::GEMM;
    } else {
        convolution_hint = ConvolutionMethodHint::DIRECT;
    }

    if (backend == "opencl") {
        target_hint = TargetHint::OPENCL;
    } else {
        target_hint = TargetHint::NEON;
    }

    if (dtype == "float32") {
        type = DataType::F32;
    } else {
        type = DataType::F16;
    }

    Graph graph;
    graph << target_hint << convolution_hint;

    if (model == "resnet-18")
        get_resnet18(&graph, type);
    else if (model == "mobilenet")
        get_mobilenet(&graph, type);
    else if (model == "vgg-16")
        get_vgg16(&graph, type);
    else if (model == "nature-dqn")
        get_nature_dqn(&graph, type);
    else
        std::cout << "unknown model" << std::endl;

    graph.graph_init(true);

    int num_warmup, num_test;

    num_warmup = 2;
    num_test   = 100;

    // warm up
    measure(&graph, num_warmup);

    // test
    double cost = measure(&graph, num_test);
    return cost;
}


int main(int argc, const char **argv)
{
    std::string backend[] = {"opencl"};
    std::string model[] = {"nature-dqn", "mobilenet", "resnet-18", "vgg-16"};
    std::string conv_method[] = {"gemm"};
    std::string dtype[] = {"float32", "float16"};

    int n_ave_curve = 3;

    for (int i = 0; i < sizeof(backend)/sizeof(backend[0]); i++) {
        for (int j = 0; j < sizeof(model)/sizeof(model[0]); j++) {
            for (int k = 0; k < sizeof(conv_method)/sizeof(conv_method[0]); k++) {
                for (int l = 0; l < sizeof(dtype)/sizeof(dtype[0]); l++) {
                    std::vector<double> costs;

                    for (int t = 0; t < n_ave_curve; t++) {
                        double cost = run_case(backend[i], model[j], conv_method[k], dtype[l]);
                        sleep(1);

                        costs.push_back(cost);
                    }

                    log_value(backend[i], "Mali-T860", model[j] + ".B1",
                              "ARMComputeLib-" + conv_method[k] + "-" + dtype[l],
                              to_python_list(costs));
                }
            }
        }
    }

    return 0;
}

