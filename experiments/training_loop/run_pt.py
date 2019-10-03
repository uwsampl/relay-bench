"""
Based on https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb
"""
import csv
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from common import invoke_main, read_config, write_status, render_exception
from trial_util import configure_seed
from validate_config import validate

import source

def load_model(model_name):
    if model_name == 'mlp':
        return source.mnist()
    raise Exception('Unsupported model: ' + model_name)


def load_raw_data(dataset_name):
    if dataset_name == 'mnist':
        train = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
        validation = datasets.MNIST('./data',
                                    train=False,
                                    transform=transforms.ToTensor())
        return (train, validation)
    raise Exception('Unsupported dataset: ' + dataset_name)


def get_data_loader(raw_data, batch_size, shuffle):
    return torch.utils.data.DataLoader(dataset=raw_data,
                                       batch_size=batch_size,
                                       shuffle=shuffle)


def train(train_loader, model, device):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        data = data.view(-1, 28*28)
        target = target.to(device)

        # Calculate loss
        loss = model[0](data, target)


def validate_learner(validation_loader, model, device):
    criterion = nn.CrossEntropyLoss()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        data = data.view(-1, 28*28)
        output = model[1](data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[
            1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    return val_loss, int(correct.item()), len(validation_loader.dataset)


def main(config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    if 'pt' not in config['frameworks']:
        write_status(output_dir, True, 'PyTorch not run')
        return 0

    configure_seed(config)
    device = source.device
    dev = 'gpu' # TODO ensure we can set this appropriately in SMLL

    batch_size = config['batch_size']
    epochs = config['epochs']
    models = config['models']
    datasets = config['datasets']
    dry_run = config['dry_run']
    reps = config['reps']

    # record: dev, model, dataset, rep, epoch, time, loss, num correct, total
    fieldnames = ['device', 'model', 'dataset', 'rep', 'epoch',
                  'time', 'loss', 'correct', 'total']
    try:
        info = []
        for dataset in datasets:
            raw_train, raw_validation = load_raw_data(dataset)
            for model_name in models:
                for rep in range(reps):
                    training = get_data_loader(raw_train, batch_size, True)
                    model = load_model(model_name)

                    # dry run: train and throw away
                    for dry_epoch in range(dry_run):
                        train(training, model, device)

                    # reload to reset internal state
                    model = load_model(model_name)
                    training = get_data_loader(raw_train, batch_size, True)
                    validation = get_data_loader(raw_validation, batch_size, False)
                    for epoch in range(epochs):
                        e_start = time.time()
                        train(training, model, device)
                        e_end = time.time()

                        e_time = e_end - e_start
                        loss, correct, total = validate_learner(
                            validation, model, device)
                        info.append((dev, model_name, dataset, rep, epoch,
                                     e_time, loss, correct, total))
                        print(model_name, dataset, rep, epoch,
                              e_time, loss, '{}/{}'.format(correct, total))
                    time.sleep(4)

        # dump to CSV
        filename = os.path.join(output_dir, 'pt-train.csv')
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in info:
                writer.writerow({
                    fieldnames[i]: row[i]
                    for i in range(len(fieldnames))
                })
    except Exception as e:
        write_status(output_dir, False,
                     'Encountered exception: {}'.format(render_exception(e)))
        return 1

    write_status(output_dir, True, 'success')
    return 0


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'output_dir')
