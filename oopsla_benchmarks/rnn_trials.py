import cProfile
import string
import time
import math
import argparse
import csv
import numpy as np

import tvm_relay.util as relay
import pytorch.util as pt
import mx.util as mx
from util import run_trials


# TODO(weberlo): Refactor this into a `main` method that takes a config dict,
# and in the cmd-line case, argparse generates that dict.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get hidden')
    parser.add_argument('--n-hidden', type=int, default=16,
                        help='Number of hidden layers')
    parser.add_argument("--output-dir", type=str, default='')
    parser.add_argument('--dry-run', type=int, default=8)
    parser.add_argument('--n-times-per-input', type=int, default=1000)
    parser.add_argument('--skip-relay', action='store_true')
    parser.add_argument('--skip-pytorch', action='store_true')
    parser.add_argument('--skip-mxnet', action='store_true')
    parser.add_argument('--no-cpu', action='store_true')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--no-loop', action='store_true')
    parser.add_argument('--no-cell', action='store_true')
    parser.add_argument('--no-aot', action='store_true')
    parser.add_argument('--no-intp', action='store_true')
    parser.add_argument('--append-relay-data', action='store_true')
    parser.add_argument('--skip-char-rnn', action='store_true')
    parser.add_argument('--skip-gluon-rnns', action='store_true')
    parser.add_argument('--skip-treelstm', action='store_true')
    args = parser.parse_args()

    devices = []
    # RNNs all assume CPU for now
    # if not args.no_gpu:
    #     devices.append('gpu')
    if not args.no_cpu:
        devices.append('cpu')
    if len(devices) == 0:
        exit()

    configurations = []
    if not args.no_loop:
        configurations.append('loop')
    if not args.no_cell:
        configurations.append('cell')
    if (not args.skip_relay) and len(configurations) == 0:
        raise Exception('No configurations specified but Relay is to be run')

    methods = []
    if not args.no_aot:
        methods.append('aot')
    if not args.no_intp:
        methods.append('intp')
    if (not args.skip_relay) and len(methods) == 0:
        raise Exception('No methods for running Relay are specified but Relay is to be run')

    networks = ['char-rnn']
    inputs = [string.ascii_uppercase]
    languages = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English',
                 'French', 'German', 'Greek', 'Irish', 'Italian',
                 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian',
                 'Scottish', 'Spanish', 'Vietnamese']
    hidden_sizes = [args.n_hidden]

    gluon_networks = ['rnn', 'gru', 'lstm']
    if not args.skip_relay and not args.skip_gluon_rnns:
        run_trials('relay', 'gluon',
                   args.dry_run, args.n_times_per_input, 1,
                   relay.rnn_trial, relay.gluon_rnn_setup, relay.rnn_teardown,
                   ['network', 'device', 'method'],
                   [gluon_networks, devices, methods],
                   path_prefix=args.output_dir)

    if not args.skip_mxnet and not args.skip_gluon_rnns:
        run_trials('mx', 'gluon',
                   args.dry_run, args.n_times_per_input, 1,
                   mx.rnn_trial, mx.rnn_setup, mx.rnn_teardown,
                   ['network', 'device'],
                   [gluon_networks, devices],
                   path_prefix=args.output_dir)

    if not args.skip_pytorch and not args.skip_char_rnn:
        run_trials('pytorch', 'rnn',
                   args.dry_run, args.n_times_per_input, 1,
                   pt.rnn_trial, pt.rnn_setup, pt.rnn_teardown,
                   ['network', 'device', 'hidden_size', 'language', 'input'],
                   [networks, devices, hidden_sizes, languages, inputs],
                   path_prefix=args.output_dir)

    if not args.skip_relay and not args.skip_char_rnn:
        run_trials('relay', 'rnn',
                   args.dry_run, args.n_times_per_input, 1,
                   relay.rnn_trial, relay.rnn_setup, relay.rnn_teardown,
                   ['network', 'device', 'configuration',
                    'method', 'hidden_size', 'language', 'input'],
                   [networks, devices, configurations,
                    methods, hidden_sizes, languages, inputs],
                   append_to_csv = args.append_relay_data,
                   path_prefix=args.output_dir)


    datasets = ['dev', 'test', 'train']
    treelstm_idxs = [i for i in range(500)]
    if not args.skip_pytorch and not args.skip_treelstm:
        run_trials('pytorch', 'treelstm',
                   args.dry_run, args.n_times_per_input, 1,
                   pt.rnn_trial, pt.treelstm_setup, pt.rnn_teardown,
                   ['device', 'dataset', 'idx'],
                   [devices, datasets, treelstm_idxs],
                   path_prefix=args.output_dir)


    if not args.skip_relay and not args.skip_treelstm:
        run_trials('relay', 'treelstm',
                   args.dry_run, args.n_times_per_input, 1,
                   relay.rnn_trial, relay.treelstm_setup, relay.rnn_teardown,
                   ['device', 'method', 'dataset', 'idx'],
                   [devices, methods, datasets, treelstm_idxs],
                   path_prefix=args.output_dir)
