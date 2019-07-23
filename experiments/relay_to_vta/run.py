# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Run ResNet-18 on VTA simulator.

**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This experiment runs ResNet-18 inference on the VTA accelerator design to
perform a single ImageNet classification task.
"""
import argparse
import json
import os
import time
from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np

import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util

import vta
from vta.testing import simulator
from vta.top import graph_pack

from validate_config import validate
from common import read_json, write_json, write_status

# Name of Gluon model to compile
MODEL = 'resnet18_v1'
# The `START_PACK` and `STOP_PACK` labels indicate where to start and end the
# graph packing relay pass---in other words, where to start and finish
# offloading to VTA.
START_PACK = 'nn.max_pool2d'
STOP_PACK = 'nn.global_avg_pool2d'

# TODO(weberlo): What's the flippin' diff between `vta_env.target`, and `vta_env.TARGET`?

def init_vta_env(target):
    config_dir = os.path.join(os.environ['TVM_HOME'], 'vta', 'config')
    config_filename = 'vta_config.json'
    vta_config = read_json(config_dir, config_filename)
    vta_config['TARGET'] = target
    write_json(config_dir, config_filename, vta_config)
    return vta.get_env()


def init_remote(vta_env, config):
    if vta_env.TARGET in ['sim', 'tsim']:
        # To target the simulator, we use a local RPC session as the execution
        # remote.
        remote = rpc.LocalSession()
        return remote, None
    else:
        # Get remote from tracker node if environment variable is set.
        # To set up the tracker, you'll need to follow the "Auto-tuning
        # a convolutional network for VTA" tutorial.
        tracker_host = config.get('tracker_host', None)
        tracker_port = config.get('tracker_port', None)
        # Otherwise if you have a device you want to program directly from
        # the host, make sure you've set the variables below to the IP of
        # your board.
        device_host = config.get('pynq_rpc_host', '192.168.2.99')
        device_port = config.get('pynq_rpc_port', 9091)
        if not tracker_host or not tracker_port:
            remote = rpc.connect(device_host, device_port)
        else:
            remote = autotvm.measure.request_remote(vta_env.TARGET, tracker_host, tracker_port, timeout=10000)

        # Reconfigure the JIT runtime and FPGA.
        # You can program the FPGA with your own custom bitstream
        # by passing the path to the bitstream file instead of None.
        reconfig_start = time.time()
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
        reconfig_time = time.time() - reconfig_start
        return remote, reconfig_time


def build_resnet(remote, target, ctx, vta_env):
    """Build the inference graph runtime."""
    # Load pre-configured AutoTVM schedules.
    with autotvm.tophub.context(target):
        # Populate the shape and data type dictionary for ResNet input.
        dtype_dict = {'data': 'float32'}
        shape_dict = {'data': (vta_env.BATCH, 3, 224, 224)}

        # Get off-the-shelf gluon model and convert to Relay.
        gluon_model = vision.get_model(MODEL, pretrained=True)

        # Start frontend compilation.
        mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

        # Update shape and type dictionary.
        shape_dict.update({k: v.shape for k, v in params.items()})
        dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

        # Perform quantization in Relay.
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            relay_prog = relay.quantize.quantize(mod['main'], params=params)

        # Perform graph packing and constant folding for VTA target.
        if target.device_name == 'vta':
            assert vta_env.BLOCK_IN == vta_env.BLOCK_OUT
            relay_prog = graph_pack(
                relay_prog,
                vta_env.BATCH,
                vta_env.BLOCK_OUT,
                vta_env.WGT_WIDTH,
                start_name=START_PACK,
                stop_name=STOP_PACK)

        # Compile Relay program with AlterOpLayout disabled.
        with relay.build_config(opt_level=3, disabled_pass={'AlterOpLayout'}):
            if target.device_name == 'vta':
                with vta.build_config():
                    graph, lib, params = relay.build(
                        relay_prog, target=vta_env.target,
                        params=params, target_host=vta_env.target_host)
            else:
                graph, lib, params = relay.build(
                    relay_prog, target=target,
                    params=params, target_host=env.target_host)

        # Send the inference library over to the remote RPC server
        temp = util.tempdir()
        lib.save(temp.relpath('graphlib.o'))
        remote.upload(temp.relpath('graphlib.o'))
        lib = remote.load_module('graphlib.o')

        resnet_module = graph_runtime.create(graph, lib, ctx)
        resnet_module.set_input(**params)
        return resnet_module


def get_test_image(vta_env):
    """Retrieve and prepare test image for inference."""
    image = Image.open(os.path.join(os.path.dirname(__file__), 'cat.jpg')).resize((224, 224))
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = np.repeat(image, vta_env.BATCH, axis=0)
    return image


def run_model(resnet_module, remote, ctx, vta_env, config):
    """Perform ResNet-18 inference on an ImageNet sample and collect stats."""
    image = get_test_image(vta_env)
    # Set the module input to be `image`.
    resnet_module.set_input('data', image)

    # Perform inference and gather execution statistics.
    timer = resnet_module.module.time_evaluator(
        'run', ctx, number=config['n_times_per_input'], repeat=config['num_reps'])

    if vta_env.TARGET in ['sim', 'tsim']:
        simulator.clear_stats()
        timer()
        stats = simulator.stats()
        for key in stats:
            # Since we execute the workload many times, we need to normalize stats
            # Note that there is always one warm up run
            # Therefore we divide the overall stats by (num * rep + 1)
            divisor = config['n_times_per_input'] * config['num_reps'] + 1
            stats[key] //= divisor
    else:
        tcost = timer()
        stats = {
            'mean': tcost.mean * 1000 / env.BATCH,
            'std_dev': np.std(tcost.results) * 1000 / env.BATCH
        }

    check_results(resnet_module, remote, vta_env)
    return stats


def check_results(resnet_module, remote, vta_env):
    # Get classification results.
    tvm_output = resnet_module.get_output(
        0, tvm.nd.empty((vta_env.BATCH, 1000), 'float32', remote.cpu(0)))
    top_categories = np.argsort(tvm_output.asnumpy()[0])

    # Load ImageNet categories.
    with open(os.path.join(os.path.dirname(__file__), 'synset.txt')) as f:
        # "synset.txt" is formatted as a Python dict, so we eval the file
        # contents to get a Python dict.
        synset = eval(f.read())

    # This just checks that one of the 5 top categories is one variety of cat;
    # this is by no means an accurate assessment of how quantization affects
    # classification accuracy but is meant to catch changes to the quantization
    # pass that would accuracy in the CI.
    cat_detected = False
    for k in top_categories[-5:]:
        if 'cat' in synset[k]:
            cat_detected = True
    assert(cat_detected)


def main(config_dir, output_dir):
    """Run the experiment."""
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return

    result = {}
    for target_name in config['targets']:
        vta_env = init_vta_env(target_name)
        device = config['device']
        target = vta_env.target if device == 'vta' else vta_env.target_vta_cpu

        # Make sure TVM was compiled with RPC support.
        assert tvm.module.enabled('rpc')
        remote, reconfig_time = init_remote(vta_env, config)
        # Get execution context from remote
        ctx = remote.ext_dev(0) if device == 'vta' else remote.cpu(0)

        resnet_module = build_resnet(remote, target, ctx, vta_env)
        sim_stats = run_model(resnet_module, remote, ctx, vta_env, config)
        if reconfig_time is not None:
            sim_stats['reconfig_time'] = reconfig_time

        result[target_name] = sim_stats

    write_json(output_dir, 'data.json', result)
    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir)
