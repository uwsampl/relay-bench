import numpy as np
import torch
import argparse
import time
import tvm

import torchvision.models as models
from util.mobilenetv2 import MOBILENET_PARAMS
from util.mobilenetv2.MobileNetV2 import MobileNetV2 as mobilenet
from util.dcgan.dcgan import Generator as dcgan
from util.dcgan import DCGAN_PARAMS

def load_params(location, dev):
    if dev != 'cpu':
        return torch.load(location)
    else:
        return torch.location(location, map_location='cpu')

def instantiate_network(network, batch_size, dev):
    image_shape = (224, 3, 224, batch_size)

    if network == 'resnet-18':
        net = models.resnet18(pretrained=True)
    elif network == 'vgg-16':
        net = models.vgg16(pretrained=True)
    elif network == 'mobilenet':
        net = mobilenet(n_class=1000)
        loc = MOBILENET_PARAMS
        state_dict = load_params(MOBILENET_PARAMS, dev)
        net.load_state_dict(state_dict)
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
