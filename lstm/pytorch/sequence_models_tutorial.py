import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lstm.language_data import sentences
import time
from benchmark import avg_time_since

torch.manual_seed(1)

def prepare_sequence(idx, seq, to_ix):
    idxs = [to_ix[w[idx]] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = sentences

word_to_ix = {}
tag_to_ix = {}

def insert_ix(d, e):
    if e not in d:
        d[e] = len(d)

for sent in training_data:
    for word, tag in sent:
        insert_ix(word_to_ix, word)
        insert_ix(tag_to_ix, tag)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def bm():
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    t = time.time()
    for epoch in range(1000):
        for sentence in training_data:
            with torch.no_grad():
                model.hidden = model.init_hidden()
                sentence_in = prepare_sequence(0, sentence, word_to_ix)
                tag_scores = model(sentence_in)
    print(avg_time_since(t, 1))
