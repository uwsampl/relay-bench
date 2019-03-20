import cProfile
import string
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


def benchmark(num, trial, fieldnames, args, dry_run, iterations, writer):
    for i in range(dry_run + iterations):
        if i == dry_run:
            tic = time.time()
        trial(*args)
    final = time.time()

    write_row(writer, fieldnames, list(args) + [num, final - tic])
    return (final - tic) / iterations


def run_trials(method, task_name,
               dry_run, times_per_input, repetitions,
               trial, parameter_names, parameter_values):
    filename = '{}-{}.csv'.format(method, task_name)
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = parameter_names + ['rep', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        costs = []
        for t in range(len(parameter_values)):
            for i in range(repetitions):
                score = benchmark(i, trial, parameter_names, parameter_values[t], dry_run, times_per_input, writer)

                if t != repetitions - 1:
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
    parser.add_argument('--n-repetitions', type=int, default=100)
    parser.add_argument('--n-times-per-input', type=int, default=1000)
    parser.add_argument('--cell-only', action='store_true')
    parser.add_argument('--skip-pytorch', action='store_true')
    parser.add_argument('--use-cpu', action='store_true')
    args = parser.parse_args()

    parameter_names = ['Language', 'Starting Letters']
    parameter_values = [('Arabic', string.ascii_uppercase),
                        ('Chinese', string.ascii_uppercase),
                        ('Czech', string.ascii_uppercase),
                        ('Dutch', string.ascii_uppercase),
                        ('English', string.ascii_uppercase),
                        ('French', string.ascii_uppercase),
                        ('German', string.ascii_uppercase),
                        ('Greek', string.ascii_uppercase),
                        ('Irish', string.ascii_uppercase),
                        ('Italian', string.ascii_uppercase),
                        ('Japanese', string.ascii_uppercase),
                        ('Korean', string.ascii_uppercase),
                        ('Polish', string.ascii_uppercase),
                        ('Portuguese', string.ascii_uppercase),
                        ('Russian', string.ascii_uppercase),
                        ('Scottish', string.ascii_uppercase),
                        ('Spanish', string.ascii_uppercase),
                        ('Vietnamese', string.ascii_uppercase)]

    gpu = not args.use_cpu

    # f=open(args.FILE, "a")

    task_name = 'char-rnn-{}'.format(args.n_hidden)

    # bench_forward(args.N_HIDDEN)
    if not args.skip_pytorch:
        pytorch_rnn = pytorch.char_rnn_generator.RNN(data.N_LETTERS, args.n_hidden, data.N_LETTERS)
        pytorch_trial = lambda lang, letters: pytorch.samples(pytorch_rnn, lang, letters)
        run_trials('pytorch', task_name,
                   args.dry_run, args.n_times_per_input, args.n_repetitions,
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

    run_trials(relay_method + '-intp', task_name,
               args.dry_run, args.n_times_per_input, args.n_repetitions,
               relay_trial_intp, parameter_names, parameter_values)

    run_trials(relay_method + '-aot', task_name,
               args.dry_run, args.n_times_per_input, args.n_repetitions,
               relay_trial_aot, parameter_names, parameter_values)
