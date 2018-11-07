import torch
import random
from .. import language_data as data


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(data.ALL_CATEGORIES)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = data.ALL_CATEGORIES.index(category)
    tensor = torch.zeros(1, data.N_CATEGORIES)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, data.N_LETTERS)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][data.ALL_LETTERS.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [data.ALL_LETTERS.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(data.N_LETTERS - 1) # EOS
    return torch.LongTensor(letter_indexes)


# Sample from a category and starting letter
def sample(rnn, category, start_letter='A'):
     with torch.no_grad():  # no need to track history in sampling
         category_tensor = categoryTensor(category)
         input = inputTensor(start_letter)
         hidden = rnn.initHidden()

         output_name = start_letter

         for i in range(data.MAX_LENGTH):
             output, hidden = rnn(category_tensor, input[0], hidden)
             topv, topi = output.topk(1)
             topi = topi[0][0]
             if topi == data.N_LETTERS - 1:
                 break
             else:
                 letter = data.ALL_LETTERS[topi]
                 output_name += letter
             input = inputTensor(letter)

         return output_name
