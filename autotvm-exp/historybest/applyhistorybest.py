import os
import argparse
import numpy as np
import time
import logging
import csv

import tvm
import nnvm.compiler
import nnvm.testing
from nnvm.testing.init import Xavier

from tvm.contrib import rpc
from tvm.contrib import util as tvm_util
from tvm.contrib import graph_runtime as runtime
from tvm._ffi.base import TVMError
import nnvm.symbol as symbol

import autotvm
from autotvm.record import load_from_file, ApplyHistoryBest
from autotvm import task, fleet
from autotvm.fleet import MeasureInput, get_measure_batch
from autotvm.fleet.worker import request_remote

import rnn_cell
ITER=100
EXPERIMENT=10
SLEEP=0


#LM specific
wkls = [
    #    ('RNN.B4.L2.S1.H650.V0',      'rnn',  4, 2, 1, 650, 0),
    #    ('RNN.B4.L2.S1.H650.V10000',  'rnn',  4, 2, 1, 650, 10000),

        ('LSTM.B4.L2.S1.H650.V0',     'lstm', 4, 2, 1, 650, 0),
    #('LSTM.B4.L2.S1.H650.V10000', 'lstm', 4, 2, 1, 650, 10000),
]

def get_cell(cell_type, batch_size, hidden_size, prefix):
    if cell_type == 'rnn':
        cell = rnn_cell.RNNCell(num_hidden=hidden_size, prefix=prefix, batch_size=batch_size)
    elif cell_type == 'lstm':
        cell = rnn_cell.LSTMCell(num_hidden=hidden_size, prefix=prefix, batch_size=batch_size)
    else:
        raise RuntimeError("Invalid cell type " + cell_type)

    return cell

def get_cell_wkl(cell_wkl, dtype):
    task_name, cell_type, batch_size, num_layer, seq_len, hidden_size, voc_size = cell_wkl

    # encoder
    data = symbol.Variable('data')

    if voc_size:
        data = symbol.squeeze(data)
        weight = symbol.Variable("encoder_weight", shape=(voc_size, hidden_size))
        embed = symbol.embedding(data=data, weight=weight, input_dim=voc_size,
                                 output_dim=hidden_size, name='embed')

        shape_info = {"data": (seq_len, batch_size)}
        dtype_info = {"data": 'int32'}
        embed = embed
    else:
        embed = data
        shape_info = {"data": (seq_len, batch_size, hidden_size)}
        dtype_info = {"data": 'float32'}

    outputs = embed
    for i in range(num_layer):
        outputs = symbol.squeeze(outputs)
        prefix = 'cell_l%d_' % i
        cell = get_cell(cell_type, batch_size, hidden_size, prefix)
        if seq_len == 1:
            outputs, states = cell(inputs=outputs, states=cell.begin_state())
        else:
            outputs, states = cell.unroll(seq_len, inputs=outputs,
                                          merge_outputs=True, layout='TNC')

    if voc_size:
        outputs = symbol.reshape(outputs, shape=(-1, hidden_size))
        outputs = symbol.dense(data=outputs, units=voc_size)
        outputs = symbol.reshape(outputs, shape=(-1, seq_len, voc_size))

    if isinstance(outputs, (list, tuple)):
        output = symbol.group(outputs)
    else:
        output = outputs

    net = nnvm.graph.create(output)

    input_shapes, output_shapes = nnvm.compiler.graph_util.infer_shape(net, **shape_info)

    shape_dict = dict(zip(net.index.input_names, input_shapes))
    initializer = Xavier()

    params = {}
    for k, v in shape_dict.items():
        if k in shape_info:
            continue
        init_value = np.zeros(v).astype(dtype)
        initializer(k, init_value)
        params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))

    return net, params, shape_info, output_shapes, dtype_info


def run_lm(target, target_host, cell_wkl, dtype, n_times, opt_level):
    #nnvm.compiler.engine.clear_cache()
    net, params, input_shapes, _, dtype_info = get_cell_wkl(cell_wkl, dtype)

    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, target=target, target_host=target_host,
            shape=input_shapes, params=params, dtype=dtype_info)

    # send model and param
    remote = request_remote(target)
    if remote is None:  # local
        ctx = tvm.context(str(target))
        rlib = lib
    else:
        filename = "net.tar"
        tmp = tvm_util.tempdir()
        path_name = tmp.relpath(filename)
        lib.export_library(path_name)
        remote.upload(path_name)
        ctx = remote.context(str(target), 0)
        rlib = remote.load_module(filename)
    rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
    module = runtime.create(graph, rlib, ctx)

    # build input
    for k, shape in input_shapes.items():
        module.set_input(k, tvm.nd.array(np.random.uniform(size=shape)
                                         .astype(dtype_info[k])))
    module.set_input(**rparams)

    # evaluate
    ftimer = module.module.time_evaluator("run", ctx, n_times)
    prof_res = ftimer()
    return prof_res.mean


def run_model(model, batch_size, device_type, device_key, opt_level, history_best, layerbylayer, target, target_host):
    if model == 'lm':
        with history_best:
            res = run_lm(target, target_host, wkls[0], 'float32', 10, opt_level)
    else:
        if not layerbylayer:
            with history_best:
                #if device_key == 'gfx900':
                #    res = build_and_run_local(model, batch_size, device_type, device_key, target, opt_level)
                #else:
                res = build_and_run(model, batch_size, device_type, device_key, target, opt_level)
        else:
            res = build_and_run_lbyl(model, batch_size, device_type, device_key, history_best, target, target_host)
    return res

def build_and_run_lbyl(model, batch_size, device_type, device_key, history_best, target, target_host):
    timings = list()
    i = 1
    print("device key:", device_key)
    ex = fleet.create(device_key, mode='local')
    measure_batch = fleet.get_measure_batch(ex, target, target_host, repeat=ITER, rpc_timeout=180)
    while True:
        if model == 'resnet18':
            task_base = 'resnet'
            type_str = 'C'
        elif model == 'mobilenet':
            task_base = 'mobilenet'
            type_str = 'C'
        elif model == 'mobilenet_depthwise':
            task_base = 'mobilenet'
            type_str = 'D'
        elif model == 'vgg16':
            task_base = 'vgg'
            type_str = 'C'
        else:
            assert False, 'model not supported'
        try:
            tsk = task.name2task('{0:s}.{1:s}{2:d}.B1'.format(task_base, type_str, i))
        except RuntimeError:
            break
        # hack to get workload from task
        keys = ['vanilla', 'spatial_pack']
        for key in keys:
            try:
                tsk.init_space(target, key)
                break
            except KeyError:
                continue
        res = list()
        for e in range(EXPERIMENT):
            logging.info("{0:d} of {1:d}".format(e, EXPERIMENT))
            time.sleep(SLEEP)
            try:
                inp = MeasureInput(target, tsk, history_best.query(target, tsk.workload))
            except RuntimeError:
                continue
            res += measure_batch([inp])
        timings.append(res)
        i += 1
    return timings
        

def build_and_run(model, batch_size, device_type, device_key, target, opt_level=3, num_iter=ITER):
    url = os.environ['TVM_TRACKER_HOST']
    port = int(os.environ['TVM_TRACKER_PORT'])
    res = list()
    i = 0

    def one_experiment(i):
        print(EXPERIMENT)
        logging.info("{0:d} of {1:d}".format(i, EXPERIMENT))
        time.sleep(SLEEP)
        tracker = rpc.connect_tracker(url, port)
        print("requesting...")
        remote = tracker.request(device_key, priority=0, session_timeout=180)
        print("requested...")
        # TODO parametrize
        if device_type == "cuda":
            unroll = 1400
        else:
            unroll = 128

        ctx = remote.context(str(target), 0)

        num_classes = 1000
        image_shape = (3, 224, 224)

        data_shape = (batch_size,) + image_shape
        out_shape = (batch_size, num_classes)
        if model == 'resnet18':
            net, weights = nnvm.testing.resnet.get_workload(
                batch_size=batch_size, image_shape=image_shape)
        elif model == 'mobilenet':
            net, weights = nnvm.testing.mobilenet.get_workload(
                batch_size=batch_size, image_shape=image_shape)
        elif model == 'vgg16':
            net, weights = nnvm.testing.vgg.get_workload(
                batch_size=batch_size, image_shape=image_shape) 
        elif model == 'dqn':
            image_shape = (4, 84, 84)
            data_shape = (batch_size,) + image_shape
            num_actions = 18
            net, weights, = nnvm.testing.dqn.get_workload(
                batch_size=batch_size)
            out_shape = (batch_size, num_actions)
        else:
            raise ValueError('no benchmark prepared for {}.'.format(model))

        with nnvm.compiler.build_config(opt_level=opt_level):
            with tvm.build_config(auto_unroll_max_step=unroll,
                unroll_explicit=(device_type != "cuda")):
                graph, lib, params = nnvm.compiler.build(
                    net, target, shape={"data": data_shape}, params=weights)

        file_name = str(np.random.randint(1 << 31)) + "_tmp_module.tar"
        temp = tvm_util.tempdir()
        path = temp.relpath(file_name)

        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        lib.export_library(path)
        remote.upload(path)

        rlib = remote.load_module(file_name)
        rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

        module = runtime.create(graph, rlib, ctx)
        module.set_input('data', tvm.nd.array(data.astype("float32")))
        module.set_input(**rparams)
        if i == 0:
            module.run()

            with nnvm.compiler.build_config(opt_level=opt_level):
                with tvm.build_config():
                    graph_local, lib_local, params_local = nnvm.compiler.build(
                        net, "llvm", shape={"data": data_shape}, params=weights)

            local_context = tvm.cpu(0)
            module_local = runtime.create(graph_local, lib_local, local_context)
            module_local.set_input('data', tvm.nd.array(data.astype("float32")))
            module_local.set_input(**params_local) 
            module_local.run()
            out = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx))
            out_local = module_local.get_output(0, tvm.nd.empty(out_shape, ctx=local_context))

            if np.allclose(out.asnumpy(), out_local.asnumpy()):
                logging.info("allclose ok")
            else:
                logging.info("RESULT MISMATCH")
                logging.info("max diff: " + str(max(abs(out_local.asnumpy()[0] - out.asnumpy()[0]))) )
                logging.info(out_local.asnumpy()[0])
                logging.info(out.asnumpy()[0])
         
                logging.info(abs(out_local.asnumpy()[0] - out.asnumpy()[0]))
        else: 
            print("run...")
            ftimer = module.module.time_evaluator("run", ctx, ITER)
            try:
                res.append(ftimer())
            except TVMError as e:
                print(e)
                logging.info("retry...")
                i -= 1
        return i
    
    while i < EXPERIMENT + 1:
        i = one_experiment(i)
        i += 1
    return res
   
def load_history_best(record_dir):
    records = list()
    for dirpath, _, filenames in os.walk(record_dir):
        for filename in filenames:
            logging.info("loading... {0:s}".format(filename))
            for record in load_from_file(os.path.join(dirpath, filename)):
                records.append(record)
    history_best = ApplyHistoryBest(records)
    return history_best 


def applyhistorybest(record_dir, model, device_type, device_key, batch_size, opt_level,
                     layerbylayer, backend=None):
    if device_type == 'llvm':
        if device_key == 'rpi3b':
            target = tvm.target.rasp() 
            target_host = 'llvm -mtriple=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon'
        elif device_key == 'rk3399':
            target = tvm.target.mali("-model=Mali-T860MP4@800Mhz")
            target_host = 'llvm -target=aarch64-linux-gnu -mattr=+neon'
        elif device_key == 'gfx900':
            assert backend is not None
            target = tvm.target.create(backend + ' -model=gfx900')
            target_host = 'llvm'
        else:
            assert False, 'device key not implemented'
    elif device_type == 'cuda':
        if device_key == 'titanx':
            target = tvm.target.cuda('-model=titanx')
            target_host = 'llvm'
        else:
            assert False, 'device key not implemented'
    else:
        assert False, 'device type not implemented'


    history_best = load_history_best(record_dir)
    res = run_model(model, batch_size, device_type,
    device_key, opt_level, history_best, layerbylayer, target,
    target_host) 
    output = list()
    if not layerbylayer:
        means = list()
        for r in res:
            means.append(r.mean)
        mean = np.mean(means)
        output.append([model, 'autotvm', opt_level, mean*1e3])
        output.append([means])
    else:
        i = 1
        if 'resnet' in model:
            layer_prefix = 'C'
            model = 'resnet'
        elif 'depthwise' in model:
            model = 'mobilenet'
            layer_prefix = 'D'
        elif 'mobilenet' in model:
            layer_prefix = 'C'
            model = 'mobilenet'
            
        else:
            assert False, "layer prefix not specified yet for this model"
        for layer_result in res:
            means = list()
            for measure_result in layer_result:
                means.append(np.mean(measure_result.runs))
            mean = np.mean(means)
            layer_name = '{0:s}.{1:s}{2:d}.B{3:d}'.format(model, layer_prefix, i, batch_size)
            tsk = task.name2task(layer_name)
            tsk = list(tsk.args[2:8])
            output.append(tsk + ['autotvm' + str(opt_level)] + [mean*1e3] + [means]) 
            i += 1
    
    return output
   

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-dir', type=str, required=True)
    parser.add_argument('--model', type=str,required=True, choices=['resnet18', 'vgg16', 'mobilenet', 'mobilenet_depthwise', 'lm', 'dqn'])
    parser.add_argument('--device-type', type=str, required=True, choices=['cuda', 'llvm'])
    parser.add_argument('--backend', type=str, required=False, choices=['rocm', 'opencl'])
    parser.add_argument('--device-key', type=str, required=True, choices=['titanx', 'rpi3b', 'gfx900'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--opt-level', type=int, default=3)
    parser.add_argument('--layerbylayer', type=bool, default=False)
    args = parser.parse_args()
    
    output = applyhistorybest(args.record_dir, args.model, args.device_type,
    args.device_key, args.batch_size, args.opt_level, args.layerbylayer, args.backend)
    print(output)
    if not args.layerbylayer:
        prefix = 'e2e'
    else:
        prefix = ''
    with open(prefix + str(args.opt_level) + args.model + '_' + args.device_key + '_' + args.record_dir.split('_')[-1] + '.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in output:
            csvwriter.writerow(row)

if __name__ == '__main__':
    main()
