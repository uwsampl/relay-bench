"""
SQuAD with Static Bidirectional Encoder Representations from Transformers (BERT)

=========================================================================================

This example shows how to finetune a model with pre-trained BERT parameters with static shape for
SQuAD, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding=utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import collections
import json
import logging
import os
import random
import time
import warnings

import numpy as np
import mxnet as mx
from mxnet import gluon, nd

import sys
sys.path.append("..")

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from static_bert_qa_model import BertForQALoss, StaticBertForQA
from bert_qa_dataset import (SQuADTransform, preprocess_dataset)
from bert_qa_evaluate import get_F1_EM, predictions
from static_bert import get_model

np.random.seed(6)
random.seed(6)
mx.random.seed(6)

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(description='BERT QA example.'
                                             'We fine-tune the BERT model on SQuAD dataset.')

parser.add_argument('--only_predict',
                    action='store_true',
                    help='Whether to predict only.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.')

parser.add_argument('--bert_dataset',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    help='BERT dataset name.'
                         'options are book_corpus_wiki_en_uncased and book_corpus_wiki_en_cased.')

parser.add_argument('--pretrained_bert_parameters',
                    type=str,
                    default=None,
                    help='Pre-trained bert model parameter file. default is None')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The output directory where the model params will be written.'
                         ' default is ./output_dir')

parser.add_argument('--epochs',
                    type=int,
                    default=3,
                    help='number of epochs, default is 3')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer',
                    type=str,
                    default='bertadam',
                    help='optimization algorithm. default is bertadam(mxnet >= 1.5.0.)')

parser.add_argument('--accumulate',
                    type=int,
                    default=None,
                    help='The number of batches for '
                         'gradients accumulation to simulate large batch size. Default is None')

parser.add_argument('--lr',
                    type=float,
                    default=5e-5,
                    help='Initial learning rate. default is 5e-5')

parser.add_argument('--warmup_ratio',
                    type=float,
                    default=0.1,
                    help='ratio of warmup steps that linearly increase learning rate from '
                         '0 to target learning rate. default is 0.1')

parser.add_argument('--log_interval',
                    type=int,
                    default=50,
                    help='report interval. default is 50')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                         'Sequences longer than this will be truncated, and sequences shorter '
                         'than this will be padded. default is 384')

parser.add_argument('--doc_stride',
                    type=int,
                    default=128,
                    help='When splitting up a long document into chunks, how much stride to '
                         'take between chunks. default is 128')

parser.add_argument('--max_query_length',
                    type=int,
                    default=64,
                    help='The maximum number of tokens for the question. Questions longer than '
                         'this will be truncated to this length. default is 64')

parser.add_argument('--n_best_size',
                    type=int,
                    default=20,
                    help='The total number of n-best predictions to generate in the '
                         'nbest_predictions.json output file. default is 20')

parser.add_argument('--max_answer_length',
                    type=int,
                    default=30,
                    help='The maximum length of an answer that can be generated. This is needed '
                         'because the start and end predictions are not conditioned on one another.'
                         ' default is 30')

parser.add_argument('--version_2',
                    action='store_true',
                    help='SQuAD examples whether contain some that do not have an answer.')

parser.add_argument('--null_score_diff_threshold',
                    type=float,
                    default=0.0,
                    help='If null_score - best_non_null is greater than the threshold predict null.'
                         'Typical values are between -1.0 and -5.0. default is 0.0')

parser.add_argument('--gpu', type=str, help='single gpu id')

parser.add_argument('--seq_length',
                    type=int,
                    default=384,
                    help='The sequence length of the input')

parser.add_argument('--input_size',
                    type=int,
                    default=768,
                    help='The embedding size of the input')

parser.add_argument('--export',
                    action='store_true',
                    help='Whether to export the model.')

args = parser.parse_args()


output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

fh = logging.FileHandler(os.path.join(
    args.output_dir, 'static_finetune_squad.log'), mode='w')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)
log.addHandler(fh)

log.info(args)

model_name = args.bert_model
dataset_name = args.bert_dataset
only_predict = args.only_predict
model_parameters = args.model_parameters
pretrained_bert_parameters = args.pretrained_bert_parameters
lower = args.uncased

epochs = args.epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size
lr = args.lr
ctx = mx.gpu(int(args.gpu)) #mx.cpu() if not args.gpu else mx.gpu(int(args.gpu))

accumulate = args.accumulate
log_interval = args.log_interval * accumulate if accumulate else args.log_interval
if accumulate:
    log.info('Using gradient accumulation. Effective batch size = {}'.
             format(accumulate * batch_size))

optimizer = args.optimizer
warmup_ratio = args.warmup_ratio

version_2 = args.version_2
null_score_diff_threshold = args.null_score_diff_threshold

max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length
n_best_size = args.n_best_size
max_answer_length = args.max_answer_length

if max_seq_length <= max_query_length + 3:
    raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                     '(%d) + 3' % (max_seq_length, max_query_length))

bert, vocab = get_model(
    name=model_name,
    dataset_name=dataset_name,
    pretrained=not model_parameters and not pretrained_bert_parameters,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False,
    input_size=args.input_size,
    seq_length=args.seq_length)

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'))

berttoken = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)


###############################################################################
#                              Hybridize the model                            #
###############################################################################
net = StaticBertForQA(bert=bert)
if pretrained_bert_parameters and not model_parameters:
    bert.load_parameters(pretrained_bert_parameters, ctx=ctx,
                         ignore_extra=True)
if not model_parameters:
    net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
else:
    net.load_parameters(model_parameters, ctx=ctx)
net.hybridize(static_alloc=True, static_shape=True)

loss_function = BertForQALoss()
loss_function.hybridize(static_alloc=True, static_shape=True)

###############################################################################
#                              Export the model                               #
###############################################################################
if __name__ == '__main__':
    import numpy as np
    from mxnet import nd
    # (24, 384) (24, 384) (24,)
    input = nd.array(np.random.rand(24, 384))
    token_types = nd.array(np.random.rand(24, 384))
    valid_length = nd.array(np.random.rand(24))
    r = net(input, token_types, valid_length)
    net.export(os.path.join(args.output_dir, 'bert_static'), epoch=args.epochs)
