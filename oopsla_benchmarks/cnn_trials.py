import argparse
import tf.util as tf
import pytorch.util as pt
import tvm_relay.util as relay
import tvm_nnvm.util as nnvm
from util import run_trials

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-inputs", type=int, default=3)
    parser.add_argument('--n-times-per-input', type=int, default=1000)
    parser.add_argument('--dry-run', type=int, default=8)
    parser.add_argument('--skip-tf', action='store_true')
    parser.add_argument('--skip-pytorch', action='store_true')
    parser.add_argument('--skip-relay', action='store_true')
    parser.add_argument('--skip-nnvm', action='store_true')
    parser.add_argument('--no-cpu', action='store_true')
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    task_name = 'cnn'
    networks = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16'] #'dcgan']

    devices = []
    if not args.no_gpu:
        devices.append('gpu')
    if not args.no_cpu:
        devices.append('cpu')
    if len(devices) == 0:
        exit()

    batch_sizes = [1]
    nnvm_opt_levels = [3]
    relay_opt_levels = [0,1,2,3]

    if not args.skip_tf:
        run_trials('tf', task_name,
                   args.dry_run, args.n_times_per_input, args.n_inputs,
                   tf.cnn_trial, tf.cnn_setup, tf.cnn_teardown,
                   ['network', 'device', 'batch_size', 'enable_xla'],
                   [networks, devices, batch_sizes, [False, True]])

    if not args.skip_pytorch:
        run_trials('pytorch', task_name,
                   args.dry_run, args.n_times_per_input, args.n_inputs,
                   pt.cnn_trial, pt.cnn_setup, pt.cnn_teardown,
                   ['network', 'device', 'batch_size'],
                   [networks, devices, batch_sizes])

    if not args.skip_relay:
        run_trials('relay', task_name,
                   args.dry_run, args.n_times_per_input, args.n_inputs,
                   relay.cnn_trial, relay.cnn_setup, relay.cnn_teardown,
                   ['network', 'device', 'batch_size', 'opt_level'],
                   [networks, devices, batch_sizes, relay_opt_levels])

    if not args.skip_nnvm:
        run_trials('nnvm', task_name,
                   args.dry_run, args.n_times_per_input, args.n_inputs,
                   nnvm.cnn_trial, nnvm.cnn_setup, nnvm.cnn_teardown,
                   ['network', 'device', 'batch_size', 'opt_level'],
                   [networks, devices, batch_sizes, nnvm_opt_levels])
