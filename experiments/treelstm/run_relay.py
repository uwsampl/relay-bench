import argparse
import sys

import torch
import tvm
from tvm import relay

import aot

from validate_config import validate
from common import write_status
from trial_util import run_trials

from run_pt import initialize_treelstm
from relay_tlstm import converter

def treelstm_setup(device, method, dataset, idx):
    use_aot = (method == 'aot')
    use_gpu = (device == 'gpu')
    torch_cpu = torch.device('cpu')
    model, data = initialize_treelstm(dataset)
    model.to(torch_cpu)
    model.eval()

    ltree, linput, rtree, rinput, label = data[idx]
    linput, rinput = linput.to(torch.device('cpu')), rinput.to(torch.device('cpu'))
    linput = model.emb(linput)

    tlstm, mod, prelude = converter.initialize_tlstm(300, 150)

    rosetree = converter.forward(ltree, linput)
    relay_tree = converter.from_tree(prelude,
                                     rosetree.fmap(converter.pytorch_to_relay),
                                     relay.TensorType([], dtype='float32'))

    context = tvm.gpu(0) if use_gpu else tvm.cpu(0)
    target = tvm.target.cuda() if use_gpu else tvm.target.create('llvm')

    if use_aot:
        mod['main'] = tlstm.get()
        func = aot.compile(tlstm.get(), mod, ctx=context, tgt=target)
    else:
        opts = relay.transform.Sequential([relay.transform.SimplifyInference(),
                                           relay.transform.FuseOps()])
        mod['main'] = tlstm.get()
        opts(mod)
        executor = relay.create_executor(mod=mod, ctx=context, target=target)
        func = executor.evaluate()

    thunk = lambda: func(relay_tree)
    return [thunk]


def treelstm_trial(thunk):
    return thunk()


def treelstm_teardown(thunk):
    pass


def main(config_dir, output_dir, method, dataset):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'relay' not in config['frameworks']:
        write_status(output_dir, True, 'Relay not run')
        sys.exit(0)

    if method not in config['relay_methods']:
        write_status(output_dir, True, '{} not run'.format(method))
        sys.exit(0)

    datasets = config['datasets']
    max_idx = -1
    for pair in config['datasets']:
        if pair[0] == dataset:
            max_idx = pair[1]
            break

    # dataset is not included in the config, so skip
    if max_idx == -1:
        write_status(output_dir, True, 'Dataset {} not run'.format(dataset))
        sys.exit(0)

    success, msg = run_trials(
        'relay', 'treelstm',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        treelstm_trial, treelstm_setup, treelstm_teardown,
        ['device', 'method', 'dataset', 'idx'],
        [config['devices'], [method],
         [dataset], [i for i in range(max_idx)]],
        path_prefix=output_dir,
        append_to_csv=True)
    if not success:
        write_status(output_dir, success, msg)
        sys.exit(1)
    write_status(output_dir, True, 'success')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir, args.method, args.dataset)
