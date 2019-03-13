import numpy as np
import torch
import argparse
import time
import tvm

import torchvision.models as models
from util.mobilenetv2 import PRETRAIN_PARAMS as MOBILENET_PARAMS
from util.mobilenetv2.MobileNetV2 import MobileNetV2

def instantiate_network(network, batch_size, dev):
    image_shape = (224, 3, 224, batch_size)

    if network == 'resnet-18':
        net = models.resnet18(pretrained=True)
    elif network == 'vgg-16':
        net = models.vgg16(pretrained=True)
    elif network == 'mobilenet-v2':
        net = MobileNetV2(n_class=1000)
        loc = MOBILENET_PARAMS
        state_dict = torch.load(loc) if dev != 'cpu' else torch.load(loc, map_location='cpu')
        net.load_state_dict(state_dict)

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
