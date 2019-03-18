import argparse
import tvm_relay.util.util as relay
from util import run_trials

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-inputs", type=int, default=3)
    parser.add_argument('--n-times-per-input', type=int, default=1000)
    parser.add_argument('--dry-run', type=int, default=8)
    parser.add_argument('--no-cpu', action='store_true')
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    task_name = 'import'
    mxnet_networks = ['resnet-18', 'nature-dqn', 'vgg-16', 'dcgan']
    onnx_networks = ['resnet-18', 'vgg-16', 'mobilenet']
    keras_networks = ['vgg-16', 'mobilenet']

    devices = []
    if not args.no_gpu:
        devices.append('gpu')
    if not args.no_cpu:
        devices.append('cpu')
    if len(devices) == 0:
        exit()

    opt_levels = [3]

    batch_sizes = [1]

    run_trials('mxnet', task_name,
               args.dry_run, args.n_times_per_input, args.n_inputs,
               relay.cnn_trial, relay.mxnet_setup, relay.cnn_teardown,
               ['network', 'device', 'batch_size', 'opt_level'],
               [mxnet_networks, devices, batch_sizes, opt_levels])

    run_trials('onnx', task_name,
               args.dry_run, args.n_times_per_input, args.n_inputs,
               relay.cnn_trial, relay.onnx_setup, relay.cnn_teardown,
               ['network', 'device', 'batch_size', 'opt_level'],
               [onnx_networks, devices, batch_sizes, opt_levels])

    run_trials('keras', task_name,
               args.dry_run, args.n_times_per_input, args.n_inputs,
               relay.cnn_trial, relay.keras_setup, relay.cnn_teardown,
               ['network', 'device', 'batch_size', 'opt_level'],
               [keras_networks, devices, batch_sizes, opt_levels])
