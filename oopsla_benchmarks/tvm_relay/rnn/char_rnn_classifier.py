import requests
import os, zipfile
import glob
import unicodedata
import string
import numpy as np
import torch
from torch import nn
from tvm import relay
from tvm.relay import op

DATA_URL = 'https://download.pytorch.org/tutorial/data.zip'
DATA_PATH = 'data'

ALL_LETTERS = string.ascii_letters + " .,;'"
NUM_LETTERS = len(ALL_LETTERS)
LETTER_TO_INDEX = dict((c, ALL_LETTERS.find(c)) for c in ALL_LETTERS)
NUM_HIDDEN = 128

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def get_data():
    if not os.path.exists(DATA_PATH):
        resp = requests.get(DATA_URL)
        with open('data.zip', 'wb') as zip_file:
            zip_file.write(resp.content)

        zip_file = zipfile.ZipFile('data.zip')
        zip_file.extractall('.')

    languages = {}
    for language in glob.glob(os.path.join(DATA_PATH , 'names', "*")):
        with open(language, encoding='utf-8') as language_file:
            category = os.path.splitext(os.path.basename(language))[0]
            lines = language_file.read().strip().split('\n')
            names = [unicode_to_ascii(line) for line in lines]
            languages[category] = names

    return languages

# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = np.zeros((1, NUM_LETTERS), 'float32')
    tensor[0][letter_to_index(letter)] = 1
    return tensor

# # Turn a line into a <line_length x 1 x n_letters>,
# # or an array of one-hot letter vectors
# def lineToTensor(line):
#     tensor = torch.zeros(len(line), 1, n_letters)
#     for li, letter in enumerate(line):
#         tensor[li][0][letterToIndex(letter)] = 1
#     return tensor

# print(letterToTensor('J'))

# print(lineToTensor('Jones').size())

def linear(input_size, output_size, x):
    weight = relay.var('linear_weight', shape=(output_size, input_size))
    return op.nn.dense(x, weight)


def rnn_cell(input_size, hidden_size, output_size):
    inp = relay.var('input')
    hidden = relay.var('hidden')
    combined = op.concatenate([inp, hidden], axis=1)
    hidden = linear(input_size + output_size, hidden_size, combined)
    output = linear(input_size + hidden_size, output_size, combined)
    output = op.nn.log_softmax(output, axis=1)
    body = relay.Tuple([output, hidden])
    return relay.Function(relay.ir_pass.free_vars(body), body)

def main():
    names_by_language = get_data()
    n_categories = len(names_by_language)
    inp = letter_to_tensor('A')
    rnn = rnn_cell(NUM_LETTERS, NUM_HIDDEN, n_categories)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
