import csv
import numpy as np
import os

from validate_config import validate
from common import invoke_main, write_status, write_json, render_exception


def data_file(data_dir, fw):
    return os.path.join(data_dir, '{}-train.csv'.format(fw))


def compute_summary(data):
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data)
    }


def main(data_dir, config_dir, output_dir):
    config, msg = validate(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    frameworks = config['frameworks']
    devices = config['devices']
    epochs = config['epochs']
    datasets = config['datasets']
    models = config['models']

    fieldnames = ['device', 'model', 'dataset', 'rep', 'epoch',
                  'time', 'loss', 'correct', 'total']

    listing_settings = {
        'PyTorch': 'pt'
    }

    # report final accuracy, final loss, average time per epoch across reps
    ret = {}
    try:
        for dev in devices:
            ret[dev] = {}
            for listing, spec_settings in listing_settings.items():
                ret[dev][listing] = {dataset: {model: {} for model in models}
                                     for dataset in datasets}
                fw = spec_settings

                epoch_times = {dataset: {model: [] for model in models}
                               for dataset in datasets}
                final_accs = {dataset: {model: [] for model in models}
                              for dataset in datasets}
                final_losses = {dataset: {model: [] for model in models}
                                for dataset in datasets}

                filename = data_file(data_dir, fw)
                with open(filename, newline='') as csvfile:
                    reader = csv.DictReader(csvfile, fieldnames)
                    for row in reader:
                        if row['device'] != dev:
                            continue
                        epoch_times[row['dataset']][row['model']].append(
                            float(row['time']))
                        if int(row['epoch']) == epochs - 1:
                            final_accs[row['dataset']][row['model']].append(
                                float(row['correct'])/float(row['total']))
                            final_losses[row['dataset']][row['model']].append(
                                float(row['loss']))

            for dataset in datasets:
                for model in models:
                    ret[dev][listing][dataset][model] = {
                        'time': compute_summary(epoch_times[dataset][model]),
                        'acc': compute_summary(final_accs[dataset][model]),
                        'loss': compute_summary(final_losses[dataset][model])
                    }

        write_json(output_dir, 'data.json', ret)
        write_status(output_dir, True, 'success')

    except Exception as e:
        write_status(output_dir, False, render_exception(e))
        return 1


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
