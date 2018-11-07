import numpy as np
import rnn.language_data as data
import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(data.ALL_CATEGORIES)
    line = randomChoice(data.__DATA__[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = data.ALL_CATEGORIES.index(category)
    tensor = np.zeros((1, data.N_CATEGORIES))
    tensor[0][li] = 1
    return tensor.astype('float32')

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = np.zeros((len(line), data.N_LETTERS))
    for li in range(len(line)):
        letter = line[li]
        tensor[li][data.ALL_LETTERS.find(letter)] = 1
    return tensor.astype('float32')

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [data.ALL_LETTERS.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(data.N_LETTERS - 1) # EOS
    return np.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def random_training_example():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def sample(rnn, category, start_letter='A'):
    category_tensor = categoryTensor(category)
    input = data.letter_to_topi(start_letter)
    hidden = rnn.hidden
    output_name = start_letter
    for i in range(data.MAX_LENGTH):
        output, hidden, input, b = rnn(category_tensor, input, hidden)
        if b.asnumpy():
            break
        else:
            letter = data.topi_to_letter(input.data.asnumpy())
            output_name += letter
    return output_name
