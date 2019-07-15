import argparse
import os
import sys

import torch

from validate_config import validate
from common import write_status
from trial_util import run_trials

from pt_tlstm.preprocess import preprocess
from pt_tlstm.model import SimilarityTreeLSTM

def initialize_treelstm(dataset):
    tlstm_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pt_tlstm')

    emb, vocab_size, dev_data, test_data, train_data = preprocess(os.path.join(tlstm_dir, 'data/sick/'),
                                                                  os.path.join(tlstm_dir, 'data/glove/'),
                                                                  5)
    model = SimilarityTreeLSTM(vocab_size, 300, 150, 50, 5, False, False)
    model.emb.weight.data.copy_(emb)

    if dataset == 'dev':
        data = dev_data
    elif dataset == 'test':
        data = test_data
    elif dataset == 'train':
        data = train_data
    return model, data


def treelstm_setup(device, dataset, idx):
    device = torch.device('cpu')

    model, data = initialize_treelstm(dataset)
    model.to(device)
    model.eval()

    ltree, linput, rtree, rinput, label = data[idx]
    linput, rinput = linput.to(device), rinput.to(device)
    linput = model.emb(linput)

    thunk = lambda: model(ltree, linput, rtree, rinput)
    return [thunk]


def treelstm_trial(thunk):
    return thunk()


def treelstm_teardown(thunk):
    pass


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'pt' not in config['frameworks']:
        write_status(output_dir, True, 'PT not run')
        sys.exit(0)

    datasets = config['datasets']
    for dataset, max_idx in datasets:
        success, msg = run_trials(
            'pt', 'treelstm',
            config['dry_run'], config['n_times_per_input'], config['n_inputs'],
            treelstm_trial, treelstm_setup, treelstm_teardown,
            ['device', 'dataset', 'idx'],
            [config['devices'], [dataset], [i for i in range(max_idx)]],
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
    args = parser.parse_args()
    main(args.config_dir, args.output_dir)
