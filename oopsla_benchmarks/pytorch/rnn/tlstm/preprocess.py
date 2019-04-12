from __future__ import division
from __future__ import print_function

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from .Constants import PAD_WORD, BOS_WORD, UNK_WORD, EOS_WORD 
from .vocab import Vocab
from .dataset import SICKDataset
from .utils import *


# MAIN BLOCK
def preprocess(datapath, glove, num_classes):

    train_dir = os.path.join(datapath, 'train/')
    dev_dir = os.path.join(datapath, 'dev/')
    test_dir = os.path.join(datapath, 'test/')

    # write unique words from all token files
    sick_vocab_file = os.path.join(datapath, 'sick.vocab')
    if not os.path.isfile(sick_vocab_file):
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files = token_files_a + token_files_b
        sick_vocab_file = os.path.join(datapath, 'sick.vocab')
        build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file,
                  data=[PAD_WORD, UNK_WORD,
                        BOS_WORD, EOS_WORD])

    # load SICK dataset splits
    train_file = os.path.join(datapath, 'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(train_dir, vocab, num_classes)
        torch.save(train_dataset, train_file)
    dev_file = os.path.join(datapath, 'sick_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(dev_dir, vocab, num_classes)
        torch.save(dev_dataset, dev_file)
    test_file = os.path.join(datapath, 'sick_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SICKDataset(test_dir, vocab, num_classes)
        torch.save(test_dataset, test_file)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(datapath, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(
            os.path.join(glove, 'glove.840B.300d'))
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=torch.device('cpu'))
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([PAD_WORD, UNK_WORD,
                                    BOS_WORD, EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)

    return emb, vocab.size(), dev_dataset, test_dataset, train_dataset
