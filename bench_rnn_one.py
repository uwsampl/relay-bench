import cProfile
import time
import math
from rnn import pytorch, language_data as data, relay
from benchmark import avg_time_since
import argparse
import csv
import numpy as np


def flush(name, old_time):
    f.write(f'{name},pipsqueak,{args.N_HIDDEN},{round((time.time() - old_time)*1000)}\n')


def write_row(writer, fieldnames, fields):
    record = {}
    for i in range(len(fieldnames)):
        record[fieldnames[i]] = fields[i]
    writer.writerow(record)


def benchmark(trial, fieldnames, args, dry_run, iterations, writer):
    for i in range(dry_run + iterations):
        start = time.time()
        trial(*args)
        end = time.time()
        if i < dry_run:
            tic = time.time()
            continue
        write_row(writer, fieldnames, list(args) + [i - dry_run, end - start])
    final = time.time()
    return (final - tic) / iterations


def run_trials(method, task_name,
               dry_run, times_per_input,
               trial, parameter_names, parameter_values):
    filename = '{}-{}.csv'.format(method, task_name)
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = parameter_names + ['rep', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        costs = []
        for t in range(len(parameter_values)):
            score = benchmark(trial, parameter_names, parameter_values[t], dry_run, times_per_input, writer)

            if t != len(parameter_values) - 1:
                time.sleep(4)
            costs.append(score)

        print(method, task_name, parameter_values, ["%.6f" % x for x in costs])


iterations=1
def bench_forward(hidden_size):
    for aot in [True, False]:
        gpu = False
        aot_str = 'compile' if aot else 'interpret'
        gpu_str = 'gpu' if gpu else 'cpu'
        #relay_rnn = relay.char_rnn_generator.RNNCellOnly(aot, gpu, data.N_LETTERS, hidden_size, data.N_LETTERS)
        for i in range(iterations):
            break
            relay.samples(relay_rnn, 'Russian', 'RUS')
            t = time.time()
            relay.samples(relay_rnn, 'Russian', 'RUS')
            relay.samples(relay_rnn, 'German', 'GER')
            relay.samples(relay_rnn, 'Spanish', 'SPA')
            relay.samples(relay_rnn, 'Chinese', 'CHI')
            flush(f'relay_cell_{aot_str}_{gpu_str}', t)

        relay_rnn = relay.char_rnn_generator.RNNLoop(aot, gpu, data.N_LETTERS, hidden_size, data.N_LETTERS)
        for i in range(iterations):
            relay_rnn.samples('Russian', 'RUS')
            t = time.time()
            relay_rnn.samples('Russian', 'RUS')
            relay_rnn.samples('German', 'GER')
            relay_rnn.samples('Spanish', 'SPA')
            relay_rnn.samples('Chinese', 'CHI')
            flush(f'relay_loop_{aot_str}_{gpu_str}', t)

    pytorch_rnn = pytorch.char_rnn_generator.RNN(data.N_LETTERS, hidden_size, data.N_LETTERS)
    for i in range(iterations):
        pytorch.samples(pytorch_rnn, 'Russian', 'RUS')
        t = time.time()
        pytorch.samples(pytorch_rnn, 'Russian', 'RUS')
        pytorch.samples(pytorch_rnn, 'German', 'GER')
        pytorch.samples(pytorch_rnn, 'Spanish', 'SPA')
        pytorch.samples(pytorch_rnn, 'Chinese', 'CHI')
        flush('pytorch', t)


def trial(network, samples):
    return lambda lang, letters: samples(network, lang, letters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get hidden')
    parser.add_argument('--n-hidden', type=int, default=16,
                        help='Number of hidden layers')
    parser.add_argument('--dry-run', type=int, default=8)
    parser.add_argument('--n-times-per-input', type=int, default=100)
    parser.add_argument('--cell-only', action='store_true')
    parser.add_argument('--skip-pytorch', action='store_true')
    parser.add_argument('--use-cpu', action='store_true')
    # parser.add_argument('FILE', type=str, nargs='?', default='rnn-data.csv',
    #                     help='csv file to append to')
    args = parser.parse_args()

    parameter_names = ['Language', 'Starting Letters']
    parameter_values = [('Arabic', 'ARA'),
                        ('Chinese', 'CHI'),
                        ('Czech', 'CZE'),
                        ('Dutch', 'DUT'),
                        ('English', 'ENG'),
                        ('French', 'FRE'),
                        ('German', 'GER'),
                        ('Greek', 'GRE'),
                        ('Irish', 'IRI'),
                        ('Italian', 'ITA'),
                        ('Japanese', 'JAP'),
                        ('Korean', 'KOR'),
                        ('Polish', 'POL'),
                        ('Portuguese', 'POR'),
                        ('Russian', 'RUS'),
                        ('Scottish', 'SCO'),
                        ('Spanish', 'SPA'),
                        ('Vietnamese', 'VIE')]

        # ('Russian', 'RUS'),
        #                 ('German', 'GER'),
        #                 ('Spanish', 'SPA'),
        #                 ('Chinese', 'CHI')]

    gpu = not args.use_cpu

    # f=open(args.FILE, "a")

    # bench_forward(args.N_HIDDEN)
    if not args.skip_pytorch:
        pytorch_rnn = pytorch.char_rnn_generator.RNN(data.N_LETTERS, args.n_hidden, data.N_LETTERS)
        pytorch_trial = lambda lang, letters: pytorch.samples(pytorch_rnn, lang, letters)
        run_trials('pytorch', 'char-rnn',
                   args.dry_run, args.n_times_per_input,
                   pytorch_trial, parameter_names, parameter_values)

    # have to structure it this way because AOT is bugged if it runs
    # more than once in a single run of Python
    if args.cell_only:
        relay_rnn_intp = relay.char_rnn_generator.RNNCellOnly(False, gpu, data.N_LETTERS,
                                                              args.n_hidden, data.N_LETTERS)
        relay_rnn_aot = relay.char_rnn_generator.RNNCellOnly(True, gpu, data.N_LETTERS,
                                                             args.n_hidden, data.N_LETTERS)
    else:
        relay_rnn_intp = relay.char_rnn_generator.RNNLoop(False, gpu, data.N_LETTERS,
                                                          args.n_hidden, data.N_LETTERS)
        relay_rnn_aot = relay.char_rnn_generator.RNNLoop(True, gpu, data.N_LETTERS,
                                                         args.n_hidden, data.N_LETTERS)

    relay_trial_intp = lambda lang, letters: relay_rnn_intp.samples(lang, letters)
    relay_trial_aot = lambda lang, letters: relay_rnn_aot.samples(lang, letters)

    relay_method = 'relay' + ('-cell' if args.cell_only else '') + ('-gpu' if gpu else '')

    run_trials(relay_method + '-intp', 'char-rnn',
               args.dry_run, args.n_times_per_input,
               relay_trial_intp, parameter_names, parameter_values)

    run_trials(relay_method + '-aot', 'char-rnn',
               args.dry_run, args.n_times_per_input,
               relay_trial_aot, parameter_names, parameter_values)
