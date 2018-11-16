import nltk
nltk.download('treebank')
from nltk.corpus import treebank
sentences = treebank.tagged_sents()
sentences = sentences[0:10]
