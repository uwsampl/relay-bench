import argparse
import sys

from validate_config import validate
from common import write_status
from trial_util import run_trials, configure_seed

import beacon
from beacon.tensor import tensor, Tensor
import tvm
from tvm import relay

import keras
from keras.datasets import mnist

def load_mnist(num_classes):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    train = tensor(x_train), tensor(y_train)
    test = tensor(x_test), tensor(y_test)
    return train, test


def training_setup(device, batch_size, num_classes, epochs):
    gpu = (device == 'gpu')

    (x_train, y_train), (x_test, y_test) = load_mnist(num_classes)

    def training_loop():
        for e in range(epochs):
            w1 = beacon.randn(512, 784)
            b1 = beacon.randn(512)
            w2 = beacon.randn(512, 512)
            b2 = beacon.randn(512)
            w3 = beacon.randn(10, 512)
            b3 = beacon.randn(10)
            # print_every = 10
            # loss_sum = 0
            # iteration = 0
            for x, y in list(zip(x_train, y_train))[:10]:
                x = x.reshape(1, 784)
                y = y.reshape(1, 10)
                x = x.detach()
                y = y.detach()
                d1 = beacon.dense(x, w1)
                d1 = beacon.bias_add(d1, b1)
                relu1 = beacon.relu(d1)
                d2 = beacon.dense(relu1, w2, units=512)
                d2 = beacon.bias_add(d2, b2)
                relu2 = beacon.relu(d2)
                d3 = beacon.dense(relu2, w3, units=num_classes)
                d3 = beacon.bias_add(d3, b3)
                # loss = beacon.cross_entropy(d3, y)
                # loss = loss.backward()
                # loss_sum = loss.asnumpy() + loss_sum
                # iteration = iteration + 1
                # if iteration == print_every:
                #     print(loss_sum / print_every)
                #     loss_sum = 0
                #     iteration = 0

    return [training_loop]


def training_trial(thunk):
    return thunk()


def training_teardown(thunk):
    pass


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        sys.exit(1)

    if 'relay' not in config['frameworks']:
        write_status(output_dir, True, 'Relay not run')
        sys.exit(0)

    configure_seed(config)

    success, msg = run_trials(
        'relay', 'training_loop',
        config['dry_run'], config['n_times_per_input'], config['n_inputs'],
        training_trial, training_setup, training_teardown,
        ['device', 'batch_size', 'num_classes', 'input'],
        [config['devices'], config['batch_sizes'],
         config['num_classes'], config['epochs']],
        path_prefix=output_dir)

    write_status(output_dir, success, msg)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.config_dir, args.output_dir)
