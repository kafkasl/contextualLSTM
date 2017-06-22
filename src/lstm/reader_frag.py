# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf
import numpy as np

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("latin-1").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids_translating(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def _file_to_word_ids(filename):
    data = []
    files = open(filename).read().split()
    for f in files:
        with open(f) as fn:
            data.extend([int(w) for w in fn.read().split()])

    return data


def wiki_raw_data(data_path=None, word_to_id_path=None):
    """Load WP raw data from data directory "data_path".

    Reads WP text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The WP dataset comes from Tomas Mikolov's webpage:
e
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """
    import sys

    print("Loading data from %s" % data_path)
    train_path = os.path.join(data_path, "train.list")
    valid_path = os.path.join(data_path, "valid.list")
    test_path = os.path.join(data_path, "test.list")

    # word_to_id = VectorManager.read_vector(word_to_id_path)
    # print("Word 2 ID size: %s" % (sys.getsizeof(word_to_id)))
    # sys.stdout.flush()

    #word_to_id = _build_vocab(train_path)
    train_data = open(train_path).read().split()
    print("Train size: %s" % (len(train_data)))
    sys.stdout.flush()

    valid_data = _file_to_word_ids(valid_path)
    print("Validation size: %s" % (len(valid_data)))
    sys.stdout.flush()

    test_data = _file_to_word_ids(test_path)
    print("Test size: %s" % (len(test_data)))
    sys.stdout.flush()

    # vocabulary = len(word_to_id)

    return train_data, valid_data, test_data

def wiki_producer(data_name, raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw Wikipedia data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).

    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.

    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "WPProducer", [raw_data, batch_size, num_steps]):

        if data_name == "TrainInput":
            data_len = 1516132009  # Validated
        elif data_name == "ValidInput":
            data_len = 182828964  # Validated
        elif data_name == "TestInput":
            data_len = 181755142  # Validated
        else:
            print("[ERROR] Data length not defined.")
            data_len = 0

        stride = 3500000


        #raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        #raw_data = np.array(raw_data, dtype=np.int32)

        batch_len = data_len // batch_size
        print("Indices %s %s %s" % (batch_size * batch_len,
                          batch_size, batch_len))
        data = np.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")


        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

        index_0 = tf.multiply(i, num_steps)
        index_aux = tf.add(i, 1)
        index_1 = tf.multiply(index_aux, num_steps)
        # i2 = q.dequeue()
        # x = data[0:batch_size, i * num_steps:(i + 1) * num_steps]
        #print("Slices [0, %s], [%s, %s]" % (i * num_steps, batch_size, (i + 1) * num_steps))
        x = tf.strided_slice(data, [0, i * num_steps],
                              [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])

        y = data[0:batch_size, i * num_steps + 1:(i + 1) * num_steps + 1]
        y = tf.strided_slice(data, [0, i * num_steps + 1],
                             [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])

        return x, y, i*num_steps, (i + 1) * num_steps, data
