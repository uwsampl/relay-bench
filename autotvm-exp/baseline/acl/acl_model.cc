/*
 * End-to-End evaluation of models
 *
 * Build: g++ mali_acl_model.cc build/utils/*.o -std=c++11 -I. -Iinclude -Lbuild -Lbuild/opencl-1.2-stubs -larm_compute -larm_compute_graph -larm_compute_core -lOpenCL
 */

#include "arm_compute/graph.h"
#include "arm_compute/graph/ITensorAccessor.h"

#ifdef USE_OPENCL
#include "arm_compute/runtime/CL/CLScheduler.h"
#endif

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

using namespace arm_compute::utils;
using namespace arm_compute::graph;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;


// write result to uniform baseline log file
void log_value(std::string target, std::string backend, std::string workload_type,
               std::string workload, std::string tuner, std::string template_name, std::string value, std::string outfile="tmp.tsv") {
    std::stringstream ss;

    ss << target << "\t"
       << backend << "\t"
       << workload_type << "\t"
       << workload << "\t"
       << tuner << "\t"
       << template_name << "\t"
       << value << "\t"
       << 0;

    std::ofstream fout(outfile, std::ios::app);
    fout << ss.str() << std::endl;
    std::cout << ss.str() << std::endl;
}

// convert std::vector to python's list in string format
template <typename T>
std::string to_list(std::vector<T> array) {
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
void get_vgg16(Stream *graph, arm_compute::DataType type) {
    *graph << InputLayer(TensorDescriptor(TensorShape(224U, 224U, 3U, 1U), type), dummy())
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
          << OutputLayer(dummy());
}

void get_nature_dqn(Stream *graph, arm_compute::DataType type) {
    *graph << InputLayer(TensorDescriptor(TensorShape(84U, 84U, 4U, 1U), type), dummy())
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
          << FullyConnectedLayer(18U, dummy(), dummy())
          << OutputLayer(dummy());
}


BranchLayer get_dwsc_node(Stream *graph, const std::string &data_path, std::string &&param_path,
                          unsigned int  conv_filt,
                          PadStrideInfo dwc_pad_stride_info, PadStrideInfo conv_pad_stride_info)
{
    std::string total_path = "/cnn_data/mobilenet_v1_model/" + param_path + "_";
    SubStream    sg(*graph);
    sg << DepthwiseConvolutionLayer(
                   3U, 3U, dummy(),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                   dwc_pad_stride_info)
       << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.001f)
       << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f))
       << ConvolutionLayer( 1U, 1U, conv_filt, dummy(),
                   std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr), conv_pad_stride_info)
       << BatchNormalizationLayer( dummy(), dummy(), dummy(), dummy(), 0.001f)
       << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));

    return BranchLayer(std::move(sg));
}


void get_mobilenet(Stream *graph, arm_compute::DataType type) {
    std::string data_path; /* Path to the trainable data */

    *graph << InputLayer(TensorDescriptor(TensorShape(224U, 224U, 3U, 1U), type), dummy())
          << ConvolutionLayer( 3U, 3U, 32U, dummy(),
              std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
              PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR))
          << BatchNormalizationLayer(dummy(), dummy(), dummy(), dummy(), 0.001f)
          << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, 6.f));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_1", 64, PadStrideInfo(1, 1, 1, 1), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_2", 128, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_3", 128, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_4", 256, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_5", 256, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_6", 512, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_7", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_8", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_9", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_10", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_11", 512, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_12", 1024, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << get_dwsc_node(graph, data_path, "Conv2d_13", 1024, PadStrideInfo(1, 1, 1, 1, 1, 1, DimensionRoundingType::FLOOR), PadStrideInfo(1, 1, 0, 0));
          *graph << PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
          << ConvolutionLayer( 1U, 1U, 1000U, dummy(), dummy(), PadStrideInfo(1, 1, 0, 0))
          << ReshapeLayer(TensorShape(1000U))
          << SoftmaxLayer()
          << OutputLayer(dummy());
}

void add_residual_unit(Stream *graph, int num_filter, int stride, bool dim_match) {
    SubStream right(*graph);
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
        SubStream left(*graph);
        *graph << BranchLayer(BranchMergeMethod::ADD, std::move(left), std::move(right));
    } else {
        SubStream left(*graph);
        left  << ConvolutionLayer(1U, 1U, num_filter, dummy(),
                                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                                  PadStrideInfo(stride, stride, 1, 1));
        *graph << BranchLayer(BranchMergeMethod::ADD, std::move(left), std::move(right));
    }
}


void get_resnet18(Stream *graph, arm_compute::DataType type) {
    int filter_list[] = {64, 64, 128, 256, 512};
    int num_stages = 4;
    int units[] = {2, 2, 2, 2};

    *graph << InputLayer(TensorDescriptor(TensorShape(224U, 224U, 3U, 1U), type), dummy())
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
           << OutputLayer(dummy());
}


/*
 * MEASURE FUNCTION
 */

int num_threads = 0;
double measure(Stream *graph, int n_times) {
#ifdef USE_OPENCL
    arm_compute::CLScheduler::get().default_init();
#endif
    graph->run(); graph->run();

#ifdef USE_OPENCL
    arm_compute::CLScheduler::get().sync();
#endif
    auto tbegin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_times; i++) {
        graph->run();
    }
#ifdef USE_OPENCL
    arm_compute::CLScheduler::get().sync();
#endif
    auto tend = std::chrono::high_resolution_clock::now();


    double cost = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
    return cost / n_times;
}

double run_case(std::string backend, std::string model, std::string conv_method, std::string dtype, int n_times) {
    Target            target_hint;
    ConvolutionMethod convolution_hint;
    arm_compute::DataType type;

    if (conv_method == "gemm") {
        convolution_hint = ConvolutionMethod::GEMM;
    } else if (conv_method == "direct" ) {
        convolution_hint = ConvolutionMethod::DIRECT;
    } else if (conv_method == "default") {
        convolution_hint = ConvolutionMethod::DEFAULT;
    }

    if (backend == "opencl") {
        target_hint = set_target_hint(2);
    } else {
        target_hint = set_target_hint(0);
    }

    if (dtype == "float32") {
        type = DataType::F32;
    } else {
        type = DataType::F16;
    }

    Stream graph {0, "graph"};
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
        std::cout << "unknown model: " << model << std::endl;

    GraphConfig config;
    config.use_tuner = false;
    graph.finalize(target_hint, config);

    int num_warmup, num_test;

    num_warmup = 5;
    num_test   = n_times;

    // warm up
    measure(&graph, num_warmup);

    // test
    double cost = measure(&graph, num_test);
    return cost;
}

// usage:
// ./model target

int main(int argc, const char **argv)
{
    std::string model[] = {"nature-dqn", "mobilenet", "resnet-18", "vgg-16"};
    std::string conv_method[] = {"default"};
    std::string dtype[] = {"float32", "float16"};

    if (argc != 5) {
        std::cout << "Usage: ./model device_name backend n_times n_threads" << std::endl;
        return -1;
    }
    std::string target = argv[1];
    std::string backend_type = argv[2];
    int n_times = std::atoi(argv[3]);
    int n_threads = std::atoi(argv[4]);

    arm_compute::CPPScheduler::get().set_num_threads(n_threads);

    std::vector<std::string> backend;
    if (backend_type == "all") {
#ifdef USE_OPENCL
        backend.push_back("opencl");
#endif
        backend.push_back("neon");
    } else {
        backend.push_back(backend_type);
    }

    int n_ave_curve = 3;

    for (int i = 0; i < backend.size(); i++) {
        for (int j = 0; j < sizeof(model)/sizeof(model[0]); j++) {
            for (int k = 0; k < sizeof(conv_method)/sizeof(conv_method[0]); k++) {
                for (int l = 0; l < sizeof(dtype)/sizeof(dtype[0]); l++) {
                    std::vector<double> costs;

                    // skip unsupported case
                    if ((dtype[l] == "float16" && backend[i] == "neon") ||
                        (target == "rpi3b" && model[j] == "vgg-16"))
                        continue;

                    for (int t = 0; t < n_ave_curve; t++) {
                        double cost = run_case(backend[i], model[j], conv_method[k], dtype[l], n_times);

                        if (backend[i] == "neon") {
                            sleep(10);
                        } else {
                            sleep(2);
                        }

                        costs.push_back(cost);
                    }

                    log_value(target + (backend[i] == "opencl" ? "-gpu" : "-cpu"),
                              backend[i], "network", model[j],
                              "ARMComputeLib-" + dtype[l], "default",
                              "{\"cost\": " + to_list(costs) + "}");
                }
            }
        }
    }

    return 0;
}

