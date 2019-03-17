import numpy as np
import torch
import argparse
import time
import tvm

import torchvision.models as models
from oopsla_benchmarks.pytorch.util.mobilenetv2 import MOBILENET_PARAMS
from oopsla_benchmarks.pytorch.util.mobilenetv2.MobileNetV2 import MobileNetV2 as mobilenet
from oopsla_benchmarks.pytorch.util.dcgan.dcgan import Generator as dcgan
from oopsla_benchmarks.pytorch.util.dcgan import DCGAN_PARAMS
from oopsla_benchmarks.pytorch.util.dqn.dqn import DQN as dqn
from oopsla_benchmarks.pytorch.util.dqn import DQN_PARAMS

def load_params(location, dev):
    if dev != 'cpu':
        return torch.load(location)
    else:
        return torch.load(location, map_location='cpu')

def instantiate_network(network, batch_size, dev):
    image_shape = (224, 3, 224, batch_size)

    if network == 'resnet-18':
        net = models.resnet18(pretrained=True)
    elif network == 'vgg-16':
        net = models.vgg16(pretrained=True)
    elif network == 'mobilenet':
        net = mobilenet(n_class=1000)
        state_dict = load_params(MOBILENET_PARAMS, dev)
        net.load_state_dict(state_dict)
    elif network == 'nature-dqn':
        args = lambda: None
        # params picked from default settings in Rainbow project
        args.atoms = 51
        args.history_length = 4
        args.hidden_size = 512
        args.noisy_std = 0.1
        net = dqn(args, 14)
        state_dict = load_params(DQN_PARAMS, dev)
        net.load_state_dict(state_dict)
        image_shape = (batch_size, 4, 84, 84)
    elif network == 'dcgan':
        net = dcgan(ngpu = 0 if dev == 'cpu' else 1)
        state_dict = load_params(DCGAN_PARAMS, dev)
        net.load_state_dict(state_dict)
        image_shape = (batch_size, 100, 1, 1)

    return (net, image_shape)


def evaluate_model(net, image_shape, device):
    target = net.to(device)
    input_tensor = np.random.randn(*image_shape).astype(np.float32)
    input = torch.autograd.Variable(torch.from_numpy(input_tensor))
    input = input.to(device)
    output = target(input)
    return output


def cnn_setup(network, dev, batch_size):
    net, image_shape = instantiate_network(network, batch_size, dev)
    device = torch.device('cuda' if dev == 'gpu' and torch.cuda.is_available() else 'cpu')

    target = net.to(device)
    input_tensor = np.random.randn(*image_shape).astype(np.float32)
    input = torch.autograd.Variable(torch.from_numpy(input_tensor))
    input = input.to(device)
    return [target, input]


def cnn_trial(target, input):
    return target(input)


def cnn_teardown(target, input):
    pass


def score(network, dev, batch_size, num_batches):
    net, image_shape = instantiate_network(network, batch_size, dev)

    device = torch.device('cuda' if dev == 'gpu' and torch.cuda.is_available() else 'cpu')

    dry_run = 8
    for i in range(dry_run + num_batches):
        if i == dry_run:
            tic = time.time()
        out = evaluate_model(net, image_shape, device)
        end = time.time()

    return num_batches * batch_size / (end - tic)
