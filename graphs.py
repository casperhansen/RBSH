import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np

def matrix_for_embed(w2v, mappings, maxId, size):
    '''
    Fill out the word embedding (w2v) based on the used mapping
    :param w2v: word embedding
    :param mappings: mapping from index to word (provided by a tokenizer)
    :param maxId: maximum index in mapping
    :param size: embedding size (given by the pretrained embedding, usually 2-300)
    '''
    matrix = np.random.uniform(-0.05, 0.05, (maxId, size))  # np.zeros((maxId,size[0]), dtype = np.float32)
    for word in mappings:
        wordid = mappings[word]
        if word in w2v:
            matrix[wordid, :] = w2v[word]
            print("matrix_for_embed",word, wordid)
    return matrix.astype(np.float32)


def make_embedding(vocab_size, embedding_dim, name="word_embedding", trainable=True, init=False):
    W = tf.Variable(tf.random_uniform(shape=[vocab_size, embedding_dim], minval=-0.05, maxval=0.05),
                    trainable=trainable, name=name)
    if init:
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        embedding_init = W.assign(embedding_placeholder)
        return (W, embedding_placeholder, embedding_init)
    else:
        return W, None, None


def make_importance_embedding(vocab_size, trainable=True):
    W = tf.Variable(tf.random_uniform(shape=[vocab_size], minval=0.1, maxval=1),
                    trainable=trainable, name="importance_embedding")
    return W