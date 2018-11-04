# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

from tvm import relay
from tvm.relay import op

def linear(input_size, output_size, x):
    weight = relay.var('linear_weight', shape=(output_size, input_size))
    return op.nn.dense(x, weight)

max_length = 20

import numpy as np
from tvm.relay.interpreter import evaluate
from tvm.relay.env import Environment

env = Environment()

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.fwd = relay.GlobalVar('fwd')
        self.hidden = np.zeros((1, hidden_size))
        category = relay.var('category', shape=(1, n_categories))
        inp = relay.var('input', shape=(1, input_size))
        hidden = relay.var('hidden', shape=(1, hidden_size))
        combined = op.concatenate([category, inp, hidden], axis=1)
        hidden = linear(n_categories + input_size + hidden_size, hidden_size, combined)
        output = linear(n_categories + input_size + hidden_size, output_size, combined)
        output_combined = op.concatenate([hidden, output], axis=1)
        output = linear(hidden_size + output_size, output_size, output_combined)
        # output = op.nn.dropout(output, 0.1) #attributes has not been registered
        output = op.nn.log_softmax(output, axis=1)
        body = relay.Tuple([output, hidden])
        env[self.fwd] = relay.Function(relay.ir_pass.free_vars(body), body)

    def __call__(self, category, start_letter='A'):
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = self.hidden
        output_name = start_letter
        for i in range(max_length):
            output, hidden = evaluate(env, self.fwd)
            raise
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
        return output_name

    def samples(self, category, start_letters='ABC'):
        for start_letter in start_letters:
            print(self(category, start_letter))

import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = np.zeros((1, n_categories))
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = np.zeros((len(line), 1, n_letters))
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return np.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

import time
import math

rnn = RNN(n_letters, 128, n_letters)

rnn.samples('Russian', 'RUS')

rnn.samples('German', 'GER')

rnn.samples('Spanish', 'SPA')

rnn.samples('Chinese', 'CHI')
