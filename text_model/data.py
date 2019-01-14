#!/usr/bin/python
# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf

from utils.common import *


class Vocabulary(object):
    """
    Token vocabulary.
    """
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    RESERV_TOKEN = [PAD, BOS, EOS, UNK]
    def __init__(self, vocab_file, need_reserve=True, encoding='utf-8'):
        """
        Args:
            vocab_file: a flat text file with one (normalized) token per line.
            need_reserve: if `False`, reserve-tokens will not be added.
        """
        self._id_to_word = []
        self._word_to_id = {}
        if need_reserve:
            for reserv_tok in Vocabulary.RESERV_TOKEN:
                self._word_to_id[reserv_tok] = len(self._id_to_word)
                self._id_to_word.append(reserv_tok)
        with open(vocab_file, encoding=encoding) as f:
            for line in f:
                word_name = line.strip('\n')
                self._word_to_id[word_name] = len(self._id_to_word)
                self._id_to_word.append(word_name)
        self.size = len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        else:
            return self._word_to_id[Vocabulary.UNK]

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def encode(self, sentence, add_bos_eos=False, max_len=None, split=None):
        """
        Convert a sentence to a list of ids, with special tokens added.

        Args:
            sentence: If `split` is `None`, sentence is a list of tokens.
                Else sentence is a single string with tokens separated by `split`
            add_bos_eos: If `True`, BOS/EOS will be added.
            max_len: If not `None`, the return token ids exceed `max_len` will be truncated.
        Returns:
            A numpy array of token ids for the input sentence.
        """
        if split:
            sentence = sentence.split(split)
        if add_bos_eos:
            sentence = [Vocabulary.BOS] + sentence + [Vocabulary.EOS]
        word_ids = [ self.word_to_id(word) for word in sentence ]

        return np.array(word_ids[:max_len], dtype=NP_DTYPE_INT)


def read_wembed(wembed_file):
    """
    Read word embedding from file.

    Args:
        wembed_file: file path of word embedding.

    Returns:
        Numpy array of word embeddings.
    """
    if wembed_file.find(".hdf5") != -1:
        return read_wembed_hdf5(wembed_file)
    else:
        return read_wembed_txt(wembed_file)

def read_wembed_hdf5(wembed_file, name="word_embeddings"):
    """
    Read word embedding from HDF5 file.

    Args:
        name: name of hdf5 dataset.
    """
    import h5py
    print("Reading word embeddings from hdf5 file: %s" % (wembed_file))
    with h5py.File(wembed_file, 'r') as fin:
        dataset = fin[name]
        embeddings = np.zeros([dataset.shape[0], dataset.shape[1]], dtype=NP_DTYPE)
        embeddings = dataset[...]
    print("Read word embeddings finished, vocab_size=%d, dim=%d" % \
          (embeddings.shape[0], embeddings.shape[1]))
    return embeddings

def read_wembed_txt(wembed_file, sep=" "):
    """
    Read word embedding from text file.

    Args:
        sep: seperate character within a single line.
    """
    print("Reading word embeddings from txt file: %s" % (wembed_file))
    with open(wembed_file) as fin:
        header = fin.readline().strip('\n').split(sep)
        num = int(header[0])
        dim = int(header[1])
        embeddings = np.zeros(shape=[num, dim], dtype=NP_DTYPE)
        for idx, line in enumerate(fin.readlines()):
            line = line.strip('\n')
            word_pos = line.find(sep)
            word = line[:word_pos]
            array = line[word_pos+1:]
            embeddings[idx] = np.fromstring(array, sep=sep)
    if (idx+1) != num:
        raise ValueError("Word embedding num[%d] not match with header[%d]" % (idx+1, num))
    print("Read word embeddings finished, vocab_size=%d, dim=%d" % (num, dim))
    return embeddings

