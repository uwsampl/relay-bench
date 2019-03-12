import numpy as np
import torch
import argparse
import time
import tvm

import torchvision.models as models

def instantiate_network(network, batch_size):
    image_shape = (224, 3, 224, batch_size)

    if network == 'resnet-18':
        net = models.resnet18(pretrained=True)
    elif network == 'vgg-16':
        net = models.vgg16(pretrained=True)

    return (net, image_shape)


def evaluate_model(net, image_shape, device):
    target = net.to(device)
    input_tensor = np.random.randn(*image_shape).astype(np.float32)
    input = torch.autograd.Variable(torch.from_numpy(input_tensor))
    input = input.to(device)
    output = target(input)
    return output


def score(network, dev, batch_size, num_batches):
    net, image_shape = instantiate_network(network, batch_size)

    device = torch.device('cuda' if dev == 'gpu' and torch.cuda.is_available() else 'cpu')

    dry_run = 8
    for i in range(dry_run + num_batches):
        if i == dry_run:
            tic = time.time()
        out = evaluate_model(net, image_shape, device)
        end = time.time()

    return num_batches * batch_size / (end - tic)


def log_value(device, backend, task_type, workload, method, template, value, out_file='tmp.log'):
    """
    append experiment result to a central log file

    Parameters
    ----------
    device: str
    backend: str
    task_type: str
    workload: str
    method: str
    template: str
    value: str
    out_file: str
    """
    log_line = "\t".join([str(x) for x in [device, backend, task_type, workload, method, template, value, time.time()]])
    with open(out_file, 'a') as fout:
        fout.write(log_line + "\n")


def array2str_round(x, decimal=6):
    """ print an array of float number to pretty string with round

    Parameters
    ----------
    x: Array of float or float
    decimal: int
    """
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return "[" + ", ".join([array2str_round(y, decimal=decimal)
                                for y in x]) + "]"
    format_str = "%%.%df" % decimal
    return format_str % x
