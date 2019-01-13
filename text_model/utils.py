#!/usr/bin/python
# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np

from utils.common import *


def position_embedding(seq_batch, pos_embed_size):
    """
    Generate position embedding with tensorflow, using Transformer pos_embed.

    Args:
        seq_batch: sequence batch => [batch_size, max_seq_len, embedding_size].
        pos_embed_size: dimension of position embeddings.

    Returns:
        Tensor of position embedding => [batch_size, max_seq_len, pos_embed_size].
    """
    with tf.name_scope("position_embedding"):
        assert (pos_embed_size % 2 == 0), "position embedding size must be 2x"

        batch_shape = seq_batch.get_shape().as_list()
        batch_size  = tf.shape(seq_batch)[0]
        max_seq_len = batch_shape[1]

        dim = int(pos_embed_size/2)
        embed_i = tf.range(dim, dtype=TF_DTYPE)
        embed_i = 1. / tf.pow(10000., 2 * embed_i / pos_embed_size)
        pos_j   = tf.range(max_seq_len, dtype=TF_DTYPE)
        embed_i = tf.expand_dims(embed_i, 0)
        pos_j   = tf.expand_dims(pos_j, 1)
        pos_embed = tf.matmul(pos_j, embed_i)
        sin = tf.reshape(tf.sin(pos_embed), [max_seq_len, dim, 1])
        cos = tf.reshape(tf.cos(pos_embed), [max_seq_len, dim, 1])
        pos_embed = tf.reshape(tf.concat([sin, cos], 2), [max_seq_len, pos_embed_size])
        pos_embed = tf.reshape(tf.tile(pos_embed, tf.stack([batch_size, 1])), [-1, max_seq_len, pos_embed_size])
        return pos_embed
