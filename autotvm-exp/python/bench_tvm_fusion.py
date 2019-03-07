from __future__ import absolute_import as _abs
import numpy as np
from collections import namedtuple
import math
import argparse
import csv
import time

#import topi
import tvm
from tvm.contrib import nvcc
from tvm.contrib import graph_runtime as runtime
import nnvm
from nnvm import testing

ConvWorkload = namedtuple('ConvWorkload', ['feature', 'in_filter', 'out_filter', 'kernel', 'pad', 'stride'])
DepthConvWorkload = namedtuple('DepthConvWorkload', ['feature', 'in_filter', 'channel_multiplier', 'kernel', 'pad', 'stride'])

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx

def conv_block(data, name, channels,
               kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
               epsilon=1e-5):
    """Helper function to construct conv-bn-relu"""
    # convolution + bn + relu
    conv = nnvm.sym.conv2d(data=data, channels=channels,
                           kernel_size=kernel_size, strides=strides,
                           padding=padding, use_bias=False,
                           layout="NCHW", name=name + "_conv")
    bn = nnvm.sym.batch_norm(data=conv, epsilon=epsilon, name=name + "_bn")
    act = nnvm.sym.relu(data=bn, name=name + "_relu")
    return act

def dw_conv_block(data, name, depthwise_channels,
                  kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                  epsilon=1e-5):
    """Helper function to get a separable conv block"""
    # depthwise convolution + bn + relu
    conv = nnvm.sym.conv2d(data=data, channels=depthwise_channels,
                           groups=depthwise_channels, kernel_size=kernel_size, strides=strides,
                           padding=padding, use_bias=False, layout="NCHW",
                           name=name + "_depthwise_conv")
    bn = nnvm.sym.batch_norm(data=conv, epsilon=epsilon, name=name + "_bn")
    act = nnvm.sym.relu(data=bn, name=name + "_relu")
    return act
                                                        

def bench_tvm(wkl, opt_level, num_iter):
    N = 1
    data = nnvm.sym.Variable('data')
    if isinstance(wkl, ConvWorkload):
        H, W = wkl.feature, wkl.feature
        CI = wkl.in_filter
        CO = wkl.out_filter
        HK, WK = wkl.kernel, wkl.kernel
        HPAD, WPAD = wkl.pad, wkl.pad
        HSTR, WSTR = wkl.stride, wkl.stride
        
        TH = H + 2*HPAD
        TW = W + 2*WPAD
        OH = (H + 2*HPAD - HK) // HSTR + 1
        OW = (W + 2*WPAD - WK) // WSTR + 1

        out = conv_block(data, 'test', CO, kernel_size=(HK, WK), strides=(HSTR, WSTR), padding=(HPAD, WPAD))
    else:
        H, W = wkl.feature, wkl.feature
        CI = wkl.in_filter
        CM = wkl.channel_multiplier
        CO = CI * CM
        HK, WK = wkl.kernel, wkl.kernel
        HPAD, WPAD = wkl.pad, wkl.pad
        HSTR, WSTR = wkl.stride, wkl.stride

        TH = H + 2*HPAD
        TW = W + 2*WPAD
        OH = (H + 2*HPAD - HK) // HSTR + 1
        OW = (W + 2*WPAD - WK) // WSTR + 1
        
        out = dw_conv_block(data, 'test', CO, kernel_size=(HK, WK), strides=(HSTR, WSTR), padding=(HPAD, WPAD))

    target = 'cuda'
    image_shape = (CI, H, W)
    data_shape = (N, ) + image_shape
    out_shape = (N, CO, OH, OW)

    net, params = testing.utils.create_workload(out, batch_size=N, image_shape=image_shape)
    with nnvm.compiler.build_config(opt_level=opt_level):
        with tvm.build_config(auto_unroll_max_extent=32,
                              unroll_explicit=False):
            graph, lib, params = nnvm.compiler.build(
                net, target=target, shape={"data": data_shape}, params=params)

    ctx = tvm.context(target, 0)
    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0, tvm.nd.empty(out_shape))
    out.asnumpy()

    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    prof = ftimer()
    t = round(prof.mean * 1000, 3)
    return t


resnet_workloads = [
    ConvWorkload(224,  3,  64, 7, 3, 2),
    ConvWorkload(56,  64,  64, 3, 1, 1),
    ConvWorkload(56,  64,  64, 1, 0, 1),
    ConvWorkload(56,  64, 128, 3, 1, 2),
    ConvWorkload(56,  64, 128, 1, 0, 2),
    ConvWorkload(28, 128, 128, 3, 1, 1),
    ConvWorkload(28, 128, 256, 3, 1, 2),
    ConvWorkload(28, 128, 256, 1, 0, 2),
    ConvWorkload(14, 256, 256, 3, 1, 1),
    ConvWorkload(14, 256, 512, 3, 1, 2),
    ConvWorkload(14, 256, 512, 1, 0, 2),
    ConvWorkload( 7, 512, 512, 3, 1, 1),
]


mobilenet_workloads = [
    ConvWorkload(224, 3, 32, 3, 1, 2),
    DepthConvWorkload(112, 32, 1, 3, 1, 1),
    ConvWorkload(112, 32, 64, 1, 0, 1),
    DepthConvWorkload(112, 64, 1, 3, 1, 2),
    ConvWorkload(56, 64, 128, 1, 0, 1),
    DepthConvWorkload(56, 128, 1, 3, 1, 1),
    ConvWorkload(56, 128, 128, 1, 0, 1),
    DepthConvWorkload(56, 128, 1, 3, 1, 2),
    ConvWorkload(28, 128, 256, 1, 0, 1),
    DepthConvWorkload(28, 256, 1, 3, 1, 1),
    ConvWorkload(28, 256, 256, 1, 0, 1),
    DepthConvWorkload(28, 256, 1, 3, 1, 2),
    ConvWorkload(14, 256, 512, 1, 0, 1),
    DepthConvWorkload(14, 512, 1, 3, 1, 1),
    ConvWorkload(14, 512, 512, 1, 0, 1),
    DepthConvWorkload(14, 512, 1, 3, 1, 2),
    ConvWorkload(7, 512, 1024, 1, 0, 1),
    DepthConvWorkload(7, 1024, 1, 3, 1, 1),
    ConvWorkload(7, 1024, 1024, 1, 0, 1),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, default=100, help="Number of iteration during benchmark.")
    args = parser.parse_args()

    f = open('../data/K80/conv2d_bn_relu_perf.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['network','type', 'feature', 'in_filter', 'out_filter/channel_multiplier', 'pad', 'stride', 'name', 'time(ms)'])

    def bench_net(net):
        if net == 'resnet':
            workloads = resnet_workloads
        else:
            workloads = mobilenet_workloads
        for idx, wkl in enumerate(workloads):
            print(wkl)
            for opt_level in range(2):
                t = bench_tvm(wkl, opt_level, args.num_iter)
                
                print('perf:')
                print('  opt_level: %s' % opt_level)
                print("  time: {}".format(t))
                print("")
                if isinstance(wkl, ConvWorkload):
                    row = [net, 'conv{0}x{0}'.format(wkl.kernel)] + list(wkl)[:3] + list(wkl)[4:] + [opt_level, t]
                else:
                    row = [net, 'dw_conv{0}x{0}'.format(wkl.kernel)] + list(wkl)[:3] + list(wkl)[4:] + [opt_level, t]
                writer.writerow(row)

    for net in ['mobilenet', 'resnet']:
        bench_net(net)

    f.close()

if __name__ == '__main__':
    main()
