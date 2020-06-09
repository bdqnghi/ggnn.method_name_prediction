#!/usr/bin/env/python

import numpy as np
import tensorflow as tf
import queue
import threading
import numpy as np
from sklearn import preprocessing
import os, itertools

SMALL_NUMBER = 1e-7


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)

def scale_attention_score(attention_array, top_node):
    attention_array = np.array(attention_array)
  
    top_n_idx = np.argsort(attention_array)[-top_node:][::-1]
    top_n_values = [attention_array[i] for i in top_n_idx]

    max_point = max(top_n_values)
    min_point = min(top_n_values)

    norm_data = []
    for i, point in enumerate(attention_array):
        if i in top_n_idx:
            point_norm = float((point - min_point)/(max_point - min_point))

            # Means to scale from 0 to 1
            point_scaled = point_norm * (1.0 - 0.0) + 0.0
            norm_data.append(point_scaled)
        else:
            norm_data.append(0.0)

    return norm_data


def scale_attention_score_by_group(attention_array):

    attention_array = np.array(attention_array)
    clusters = [list(g) for _, g in itertools.groupby(attention_array, lambda x: x)]
    average = float(1/len(clusters))

    scaled_attention_array = []
    for score in attention_array:
        for i, cluster in enumerate(clusters):
            if score in cluster:
                temp_score = average * (len(clusters) - i)
                scaled_attention_array.append(round(temp_score,5))

    return scaled_attention_array



class ThreadedIterator:
    """An iterator object that computes its elements in a parallel thread to be ready to be consumed.
    The iterator should *not* return None"""

    def __init__(self, original_iterator, max_queue_size: int=2):
        self.__queue = queue.Queue(maxsize=max_queue_size)
        self.__thread = threading.Thread(target=lambda: self.worker(original_iterator))
        self.__thread.start()

    def worker(self, original_iterator):
        for element in original_iterator:
            assert element is not None, 'By convention, iterator elements much not be None'
            self.__queue.put(element, block=True)
        self.__queue.put(None, block=True)

    def __iter__(self):
        next_element = self.__queue.get(block=True)
        while next_element is not None:
            yield next_element
            next_element = self.__queue.get(block=True)
        self.__thread.join()


class MLP(object):
    def __init__(self, in_size, out_size, hid_sizes, dropout_keep_prob):
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.dropout_keep_prob = dropout_keep_prob
        self.params = self.make_network_params()

    def make_network_params(self):
        dims = [self.in_size] + self.hid_sizes + [self.out_size]
        weight_sizes = list(zip(dims[:-1], dims[1:]))
        weights = [tf.Variable(self.init_weights(s), name='MLP_W_layer%i' % i)
                   for (i, s) in enumerate(weight_sizes)]
        biases = [tf.Variable(np.zeros(s[-1]).astype(np.float32), name='MLP_b_layer%i' % i)
                  for (i, s) in enumerate(weight_sizes)]

        network_params = {
            "weights": weights,
            "biases": biases,
        }

        return network_params

    def init_weights(self, shape):
        return np.sqrt(6.0 / (shape[-2] + shape[-1])) * (2 * np.random.rand(*shape).astype(np.float32) - 1)

    def __call__(self, inputs):
        acts = inputs
        for W, b in zip(self.params["weights"], self.params["biases"]):
            hid = tf.matmul(acts, tf.nn.dropout(W, self.dropout_keep_prob)) + b
            acts = tf.nn.relu(hid)
        last_hidden = hid
        return last_hidden