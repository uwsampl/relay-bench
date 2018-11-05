from io import open
import glob
import os
import string
import torch
import torch.nn as nn
from .. import language_data as data

DATA = data.get_data()

print('# categories:', data.N_CATEGORIES)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(data.N_CATEGORIES + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(data.N_CATEGORIES + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# ######################################################################
# # Training
# # =========
# # Preparing for Training
# # ----------------------
# #
# # First of all, helper functions to get random pairs of (category, line):
# #

# import random

# # Random item from a list
# def randomChoice(l):
#     return l[random.randint(0, len(l) - 1)]

# # Get a random category and random line from that category
# def randomTrainingPair():
#     category = randomChoice(all_categories)
#     line = randomChoice(category_lines[category])
#     return category, line


# ######################################################################
# # For each timestep (that is, for each letter in a training word) the
# # inputs of the network will be
# # ``(category, current letter, hidden state)`` and the outputs will be
# # ``(next letter, next hidden state)``. So for each training set, we'll
# # need the category, a set of input letters, and a set of output/target
# # letters.
# #
# # Since we are predicting the next letter from the current letter for each
# # timestep, the letter pairs are groups of consecutive letters from the
# # line - e.g. for ``"ABCD<EOS>"`` we would create ("A", "B"), ("B", "C"),
# # ("C", "D"), ("D", "EOS").
# #
# # .. figure:: https://i.imgur.com/JH58tXY.png
# #    :alt:
# #
# # The category tensor is a `one-hot
# # tensor <https://en.wikipedia.org/wiki/One-hot>`__ of size
# # ``<1 x n_categories>``. When training we feed it to the network at every
# # timestep - this is a design choice, it could have been included as part
# # of initial hidden state or some other strategy.
# #

# # One-hot vector for category
# def categoryTensor(category):
#     li = all_categories.index(category)
#     tensor = torch.zeros(1, n_categories)
#     tensor[0][li] = 1
#     return tensor

# # One-hot matrix of first to last letters (not including EOS) for input
# def inputTensor(line):
#     tensor = torch.zeros(len(line), 1, n_letters)
#     for li in range(len(line)):
#         letter = line[li]
#         tensor[li][0][all_letters.find(letter)] = 1
#     return tensor

# # LongTensor of second letter to end (EOS) for target
# def targetTensor(line):
#     letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
#     letter_indexes.append(n_letters - 1) # EOS
#     return torch.LongTensor(letter_indexes)


# ######################################################################
# # For convenience during training we'll make a ``randomTrainingExample``
# # function that fetches a random (category, line) pair and turns them into
# # the required (category, input, target) tensors.
# #

# # Make category, input, and target tensors from a random category, line pair
# def randomTrainingExample():
#     category, line = randomTrainingPair()
#     category_tensor = categoryTensor(category)
#     input_line_tensor = inputTensor(line)
#     target_line_tensor = targetTensor(line)
#     return category_tensor, input_line_tensor, target_line_tensor


# ######################################################################
# # Training the Network
# # --------------------
# #
# # In contrast to classification, where only the last output is used, we
# # are making a prediction at every step, so we are calculating loss at
# # every step.
# #
# # The magic of autograd allows you to simply sum these losses at each step
# # and call backward at the end.
# #

# criterion = nn.NLLLoss()

# learning_rate = 0.0005

# def train(category_tensor, input_line_tensor, target_line_tensor):
#     target_line_tensor.unsqueeze_(-1)
#     hidden = rnn.initHidden()

#     rnn.zero_grad()

#     loss = 0

#     for i in range(input_line_tensor.size(0)):
#         output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
#         l = criterion(output, target_line_tensor[i])
#         loss += l

#     loss.backward()

#     for p in rnn.parameters():
#         p.data.add_(-learning_rate, p.grad.data)

#     return output, loss.item() / input_line_tensor.size(0)


# ######################################################################
# # To keep track of how long training takes I am adding a
# # ``timeSince(timestamp)`` function which returns a human readable string:
# #

# import time
# import math

# def timeSince(since):
#     now = time.time()
#     s = now - since
#     m = math.floor(s / 60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)


# ######################################################################
# # Training is business as usual - call train a bunch of times and wait a
# # few minutes, printing the current time and loss every ``print_every``
# # examples, and keeping store of an average loss per ``plot_every`` examples
# # in ``all_losses`` for plotting later.
# #

# rnn = RNN(n_letters, 128, n_letters)

# n_iters = 100000
# print_every = 5000
# plot_every = 500
# all_losses = []
# total_loss = 0 # Reset every plot_every iters

# start = time.time()

# for iter in range(1, n_iters + 1):
#     output, loss = train(*randomTrainingExample())
#     total_loss += loss

#     if iter % print_every == 0:
#         print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

#     if iter % plot_every == 0:
#         all_losses.append(total_loss / plot_every)
#         total_loss = 0


# ######################################################################
# # Plotting the Losses
# # -------------------
# #
# # Plotting the historical loss from all\_losses shows the network
# # learning:
# #

# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# plt.figure()
# plt.plot(all_losses)


# ######################################################################
# # Sampling the Network
# # ====================
# #
# # To sample we give the network a letter and ask what the next one is,
# # feed that in as the next letter, and repeat until the EOS token.
# #
# # -  Create tensors for input category, starting letter, and empty hidden
# #    state
# # -  Create a string ``output_name`` with the starting letter
# # -  Up to a maximum output length,
# #
# #    -  Feed the current letter to the network
# #    -  Get the next letter from highest output, and next hidden state
# #    -  If the letter is EOS, stop here
# #    -  If a regular letter, add to ``output_name`` and continue
# #
# # -  Return the final name
# #
# # .. Note::
# #    Rather than having to give it a starting letter, another
# #    strategy would have been to include a "start of string" token in
# #    training and have the network choose its own starting letter.
# #

# max_length = 20

# # Sample from a category and starting letter
# def sample(category, start_letter='A'):
#     with torch.no_grad():  # no need to track history in sampling
#         category_tensor = categoryTensor(category)
#         input = inputTensor(start_letter)
#         hidden = rnn.initHidden()

#         output_name = start_letter

#         for i in range(max_length):
#             output, hidden = rnn(category_tensor, input[0], hidden)
#             topv, topi = output.topk(1)
#             topi = topi[0][0]
#             if topi == n_letters - 1:
#                 break
#             else:
#                 letter = all_letters[topi]
#                 output_name += letter
#             input = inputTensor(letter)

#         return output_name

# # Get multiple samples from one category and multiple starting letters
# def samples(category, start_letters='ABC'):
#     for start_letter in start_letters:
#         print(sample(category, start_letter))

# samples('Russian', 'RUS')

# samples('German', 'GER')

# samples('Spanish', 'SPA')

# samples('Chinese', 'CHI')
