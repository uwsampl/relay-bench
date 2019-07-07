import argparse
import datetime
import csv
import json
import numpy as np
import os

# info for parsing the CSVs

SCORE_FIELDS = ['rep', 'run', 'time']
BASE_FIELDS = ['network', 'device']
CNN_FIELDS = BASE_FIELDS + ['batch_size']
CHAR_RNN_FIELDS = ['hidden_size', 'language', 'input']
TREE_LSTM_FIELDS = ['dataset', 'idx']

FRAMEWORK_CNN_FIELDS = {
    'relay': CNN_FIELDS + ['opt_level'] + SCORE_FIELDS,
    'nnvm': CNN_FIELDS + ['opt_level'] + SCORE_FIELDS,
    'mxnet': CNN_FIELDS + SCORE_FIELDS,
    'tf': CNN_FIELDS + ['enable_xla'] + SCORE_FIELDS,
    'pytorch': CNN_FIELDS + SCORE_FIELDS
}

FRAMEWORK_CHAR_RNN_FIELDS = {
    'pytorch': BASE_FIELDS + CHAR_RNN_FIELDS + SCORE_FIELDS,
    'relay': BASE_FIELDS + ['configuration', 'method'] + CHAR_RNN_FIELDS + SCORE_FIELDS
}

FRAMEWORK_TREE_LSTM_FIELDS = {
    'pytorch': ['device'] + TREE_LSTM_FIELDS + SCORE_FIELDS,
    'relay': ['device', 'method'] + TREE_LSTM_FIELDS + SCORE_FIELDS
}


def lookup_data_file(data_prefix, filename):
    full_name = os.path.join(data_prefix, filename)
    if not os.path.exists(full_name):
        raise Exception('Could not find "{}"'.format(filename))
    return full_name


def prepare_out_file(output_prefix, filename):
    full_name = os.path.join(output_prefix, filename)
    if not os.path.exists(os.path.dirname(full_name)):
        os.makedirs(os.path.dirname(full_name))
    return full_name


def mean_of_means(data, trait_name, trait_values, is_numeric=False):
    means = []
    for value in trait_values:
        def filter_func(r):
            if is_numeric:
                return int(r[trait_name]) == value
            return r[trait_name] == value
        mean = np.mean(list(map(lambda r: float(r['time']),
                                filter(filter_func, data))))
        means.append(mean)
    return np.mean(means)


def average_across_reps(data, num_reps):
    return mean_of_means(data, 'rep', range(num_reps), is_numeric=True)


def framework_cnn_average(data_prefix, framework, network, dev, num_reps, opt_params):
    filename = lookup_data_file(data_prefix, '{}-cnn.csv'.format(framework))
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, FRAMEWORK_CNN_FIELDS[framework])

        def filter_func(row):
            ret = row['device'] == dev and row['network'] == network
            for (name, value) in opt_params.items():
                ret = ret and row[name] == value
                if not ret:
                    return False
            return ret
        return average_across_reps(list(filter(filter_func, reader)), num_reps)


def average_across_languages(data, languages):
    return mean_of_means(data, 'language', languages)


def framework_char_rnn_average(data_prefix, framework, network, hidden_size, languages, dev, opt_params):
    filename = lookup_data_file(data_prefix, '{}-rnn.csv'.format(framework))
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, FRAMEWORK_CHAR_RNN_FIELDS[framework])

        def filter_func(row):
            ret = row['device'] == dev and row['network'] == network
            ret = ret and int(row['hidden_size']) == hidden_size
            for (name, value) in opt_params.items():
                ret = ret and row[name] == value
                if not ret:
                    return False
            return ret
        return average_across_languages(list(filter(filter_func, reader)), languages)


def average_across_datasets(data, num_idxs, datasets):
    means = [mean_of_means(list(filter(lambda r: r['dataset'] == dataset, data)),
                           'idx', range(num_idxs), is_numeric=True)
             for dataset in datasets]
    return np.mean(means)


def framework_tree_lstm_average(data_prefix, framework, network, num_idxs, datasets, dev, opt_params):
    filename = lookup_data_file(data_prefix, '{}-treelstm.csv'.format(framework))
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, FRAMEWORK_TREE_LSTM_FIELDS[framework])

        def filter_func(row):
            ret = row['device'] == dev
            for (name, value) in opt_params.items():
                ret = ret and row[name] == value
                if not ret:
                    return False
            return ret
        return average_across_datasets(list(filter(filter_func, reader)), num_idxs, datasets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-hidden', type=int, default=16,
                        help='Number of hidden layers for char RNN')
    parser.add_argument("--data-dir", type=str, default='')
    parser.add_argument("--output-dir", type=str, default='')
    args = parser.parse_args()

    cnns = ['resnet-18', 'mobilenet', 'nature-dqn', 'vgg-16']
    num_reps = 3
    opt_levels = 4

    languages = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English',
                 'French', 'German', 'Greek', 'Irish', 'Italian',
                 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian',
                 'Scottish', 'Spanish', 'Vietnamese']
    hidden_size = args.n_hidden

    num_idxs = 500
    datasets = ['dev', 'test', 'train']

    # format:
    # {'benchmark':
    #    ({'measurement name': ('framework', opt_params)},
    #     analysis_function, networks, fixed_params)
    # }
    # for each benchmark, each measurement, each network, records:
    # {'benchmark':
    #   {'measurement':
    #     {'network': analysis_function(args.data_dir,
    #                                   framework,
    #                                   network,
    #                                   fixed_params, opt_params)}}}
    analyses = {
        'opt-cpu': ({
            'O{}'.format(i): ('relay', {'opt_level': str(i)})
            for i in range(opt_levels)
        }, framework_cnn_average, cnns, ['cpu', num_reps]),
        'opt-gpu': ({
            'O{}'.format(i): ('relay', {'opt_level': str(i)})
            for i in range(opt_levels)
        }, framework_cnn_average, cnns, ['gpu', num_reps]),
        'cnn-cpu': ({
            'Relay': ('relay', {'opt_level': str(3)}),
            'TF': ('tf', {'enable_xla': str(False)}),
            'TF XLA': ('tf', {'enable_xla': str(True)}),
            'Pytorch': ('pytorch', {}),
            'MxNet': ('mxnet', {}),
            'NNVM': ('nnvm', {})
        }, framework_cnn_average, cnns, ['cpu', num_reps]),
        'cnn-gpu': ({
            'Relay': ('relay', {'opt_level': str(3)}),
            'TF': ('tf', {'enable_xla': str(False)}),
            'TF XLA': ('tf', {'enable_xla': str(True)}),
            'Pytorch': ('pytorch', {}),
            'MxNet': ('mxnet', {}),
            'NNVM': ('nnvm', {})
        }, framework_cnn_average, cnns, ['gpu', num_reps]),
        'char-rnn': ({
            'AoT': ('relay', {'method': 'aot', 'configuration': 'loop'}),
            'Intp Cell': ('relay', {'method': 'intp', 'configuration': 'cell'}),
            'Intp Loop': ('relay', {'method': 'intp', 'configuration': 'loop'}),
            'Pytorch': ('pytorch', {})
        }, framework_char_rnn_average, ['char-rnn'], [hidden_size, languages, 'cpu']),
        'treelstm': ({
            'AoT': ('relay', {'method': 'aot'}),
            'Intp': ('relay', {'method': 'intp'}),
            'Pytorch': ('pytorch', {})
        }, framework_tree_lstm_average, ['treelstm'], [num_idxs, datasets, 'cpu'])
    }

    data = {}
    for (benchmark, (settings, summary,  networks, fixed_params)) in analyses.items():
        data[benchmark] = {}
        for (name, (framework, opt_params)) in settings.items():
            data[benchmark][name] = {
                network: summary(args.data_dir, framework, network, *fixed_params, opt_params)
                for network in networks
            }

    data['timestamp'] = str(datetime.datetime.now())

    filename = prepare_out_file(args.output_dir, 'data.json')
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
