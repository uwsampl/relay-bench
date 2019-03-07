import os
import multiprocessing
import time
import argparse
import random
import json
from importlib import import_module

import numpy as np

from util import log_value, array2str_round

def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_model(network, batch_size, dtype='float32'):
    import mxnet as mx
    if network == 'inception-v3':
        image_shape = (3,299,299) 
    elif network == 'nature-dqn':
        image_shape = (4, 84, 84)
    else:
        image_shape = (3,224,224)

    num_layers = 0
    version = 0
    if 'resnet' in network:
        num_layers = int(network.split('-')[1])
        network = 'resnet'
    if 'vgg' in network:
        num_layers = int(network.split('-')[1])
        network = 'vgg'
    if 'densenet' in network:
        num_layers = int(network.split('-')[1])
        network = 'densenet'
    if 'squeezenet' in network:
        version = network.split('_v')[1]
        network = 'squeezenet'
    if 'nature-dqn' in network:
        network = 'dqn'

    net = import_module('mx_models.' + network)

    sym = net.get_symbol(num_classes=1000,
                         image_shape = ','.join([str(i) for i in image_shape]),
                         num_layers=num_layers)
    return sym, (batch_size,) + image_shape

def run_inference(sym, arg_params, aux_params, batch_size, input_shape, num_batches,
                  use_tensorrt):
    import mxnet as mx
    mx.contrib.tensorrt.set_use_tensorrt(use_tensorrt)
    ctx = mx.gpu(0)

    data_size = input_shape
    if use_tensorrt:
        all_params = merge_dicts(arg_params, aux_params)
        executor = mx.contrib.tensorrt.tensorrt_bind(sym, ctx=ctx, all_params=all_params,
                                                     data=data_size,
                                                     softmax_label=(batch_size,),
                                                     grad_req='null',
                                                     force_rebind=True)
    else:
        executor = sym.simple_bind(ctx=ctx,
                                   data=data_size,
                                   softmax_label=(batch_size,),
                                   grad_req='null',
                                   force_rebind=True)
        executor.copy_params_from(arg_params, aux_params)

    data = mx.nd.ones(shape=data_size, ctx=ctx)

    # run
    dry_run = 10                 # use 10 iterations to warm up
    for i in range(dry_run+num_batches):
        if i == dry_run:
            tic = time.time()
        executor.forward(is_train=False, data=data)
        pred = executor.outputs[0].asnumpy()
    end = time.time()

    mx.contrib.tensorrt.set_use_tensorrt(False)

    return 1.0 / (num_batches * batch_size / (end - tic))

def score(network, batch_size, num_bathces, use_tensorrt, tmp_file):
    import tvm
    import mxnet as mx
    sym, input_shape = get_model(network, batch_size)
    ctx = mx.gpu(0)

    mod = mx.mod.Module(symbol=sym, context=ctx)
    mod.bind(data_shapes=[('data', input_shape)])
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

    arg_params, aux_params = mod.get_params()
    cost = run_inference(sym, arg_params, aux_params, batch_size, input_shape,
                         num_batches, use_tensorrt)

    device_name = tvm.gpu(0).device_name
    with open(tmp_file, 'w') as fout:
        fout.write("%f\n" % cost)
        fout.write("%s\n" % device_name)

    return cost

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()

    networks = ['resnet-18', 'resnet-50',
                'mobilenet', 'mobilenetv2',
                'vgg-16', 'vgg-19',
                'inception-v3', 'densenet-121',
                'squeezenet_v1.0', 
                'squeezenet_v1.1']

    batch_sizes = [1, 4, 8, 32]

    for b in batch_sizes:
        for use_tensorrt in [True]:
            for net in networks:
                if b == 1:
                    num_batches = 600
                elif b <= 4:
                    num_batches = 400
                elif b <= 8:
                    num_batches = 300
                elif b <= 32:
                    num_batches = 100

                if 'mobilenet' in net:
                    num_batches *= 4

                while True:
                    costs = []
                    for t in range(args.repeat):
                        tmp_file = 'cost_%0x' % random.getrandbits(32)
                        p = multiprocessing.Process(
                                target=score,
                                args=(net, b, num_batches, use_tensorrt, tmp_file))
                        p.start()
                        p.join()
                        with open(tmp_file) as fin:
                            lines = list([x.strip() for x in fin.readlines()])
                            time_cost = float(lines[0])
                            device_name = str(lines[1])
                        os.remove(tmp_file)

                        costs.append(time_cost)
                        time.sleep(2)

                    if np.std(costs) / np.mean(costs) < 0.05:
                        break
                    print(costs, "retry due to high variance in results")

                backend = 'mx-trt' if use_tensorrt else 'mx'
                print(net, b, backend, ["%.6f" % x for x in costs])

                out_file = args.out_file or device_name.replace(' ', '_') + ".tmp.tsv"
                log_value(device_name, 'cuda', 'network', "%s.B%d" % (net, b), backend,
                        'default', '{"costs" : %s}' % array2str_round(costs), out_file=out_file)

