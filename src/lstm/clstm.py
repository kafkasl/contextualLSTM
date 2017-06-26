"""
To run:

$ python lstm_frag.py --data_path=path/to/train.list

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, "../src/")

import inspect
import time
from utils.vector_manager import VectorManager
from context.creator import TopicCreator
# from context.create import get_lda_best_topic_words, get_lda_topic_embedding, get_lsa_topic_embeding
import subprocess

import numpy as np
import tensorflow as tf
from gensim.models import LsiModel, LdaModel

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

flags.DEFINE_string(
    "tasks", "all",
    "Tasks to be performed. Possible options are: all, train, test, valid")

flags.DEFINE_string(
    "word2id_path", "../models/eos/word2id_",
    "A type of model. Possible options are: small, medium, large.")

flags.DEFINE_string(
    "embeddings", "../models/eos/idWordVec_",
    "Embeddings path")

flags.DEFINE_string("topic_model_path", "../models/topics/lda_parallel_bf64b098-c517-47c8-9267-1ce116e0033d",
                    "Where the lda model is stored.")

flags.DEFINE_string("dictionary_path", "../models/topics/gensim_wordids.txt.bz2",
                    "Where the dictionary is stored.")

flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

flags.DEFINE_string("context", "lda",
                  "Type of context to be used. Possible values are, lda, lda_mean, lsi, arithmetic")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def get_context(topic_creator, segment):
    if FLAGS.context == "lda":
        return topic_creator.get_lda_best_topic_words(segment)
    if FLAGS.context == "lda_mean":
        return topic_creator.get_lda_topic_embedding(segment)
    if FLAGS.context == "lsi":
        return topic_creator.get_lsa_topic_embeding(segment)
    if FLAGS.context == "arithmetic":
        return topic_creator.average_embeddings(segment)


def generate_arrays_from_list(name, topic_creator, files, embeddings, num_steps=35, batch_size=20, embedding_size=200):
    eos_mark = [id for id, w, vec in embeddings if w == "<eos>"][0]
    eop_mark = [id for id, w, vec in embeddings if w == "<eop>"][0]
    unknown_embedding = [vec for id, w, vec in embeddings if w == "<unk>"][0]
    debug = False
    # print("EOS mark: %s, EOP mark: %s" % (eos_mark, eop_mark))
    while 1:
        for file_name in files:
            raw_list = VectorManager.parse_into_list(open(file_name).read())

            n_words = len(raw_list)
            batch_len = n_words // batch_size
            data = np.reshape(raw_list[0:batch_size*batch_len], [batch_size, batch_len])
            sentSegments = [list() for _ in range(batch_size)]
            parSegments = [list() for _ in range(batch_size)]


            for i in range(0, n_words - num_steps, 1):

                x = data[0:batch_size, i * num_steps:(i + 1) * num_steps]
                y = data[0:batch_size, i * num_steps + 1:(i + 1) * num_steps + 1]

                if len(x[0]) < num_steps or len(y[0]) < num_steps:
                    break


                emb_x = [[embeddings[int(elem)][2] for elem in l] for l in x]
                emb_x = np.reshape(emb_x, newshape=(batch_size, num_steps, embedding_size))

                final_x = np.zeros(shape=(batch_size, num_steps, len(embeddings[0][2])*3))
                for batch in range(0, batch_size):
                    for step in range(0, num_steps):
                        if debug:
                            print("%s == %s ? %s [eos]\n%s == %s ? %s[eop]" % (int(x[batch][step]), eos_mark,
                                                                           int(x[batch][step]) == eos_mark,
                                                                           int(x[batch][step]), eop_mark,
                                                                           int(x[batch][step]) == eop_mark))
                        if int(x[batch][step]) == eos_mark:
                            sentSegments[batch] = []
                        else:
                            sentSegments[batch].append(x[batch][step])
                        if int(x[batch][step]) == eop_mark:
                            parSegments[batch] = []
                        else:
                            parSegments[batch].append(x[batch][step])

                        sentTopic = unknown_embedding
                        parTopic = unknown_embedding
                        if sentSegments:
                            sentTopic = get_context(topic_creator, sentSegments[batch])

                        if parSegments:
                            if sentSegments[batch] == parSegments[batch]:
                                parTopic = sentTopic
                            else:
                                parTopic = get_context(topic_creator, parSegments[batch])

                        final_x[batch][step] = np.hstack((emb_x[batch][step], sentTopic, parTopic))



                if debug:
                    print("Batch size %s\nNum steps %s\nEmbedding size %s" % (batch_size, num_steps, embedding_size
                                                                              ))
                    print("Len(x): %s\n Len(x[0] %s\n Len(x[0][0] %s" % (len(x), len(x[0]), len(x[0][0])))
                    print("Len(y): %s\n Len(y[0] %s" % (len(y), len(y[0])))



                y = np.reshape(y, newshape=(batch_size, num_steps))

                yield final_x, y

class WPModel(object):
    """Word Prediction model."""

    def __init__(self, is_training, config):

        self.config = config
        batch_size = config.batch_size
        num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        embedding_size = config.embedding_size

        def lstm_cell():
            # With the latest TensorFlow source code (as of Mar 27, 2017),
            # the BasicLSTMCell will need a reuse parameter which is unfortunately not
            # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
            # an argument check here:
            # if 'reuse' in inspect.getargspec(
            #     tf.contrib.rnn.BasicLSTMCell.__init__).args:
            #   return tf.contrib.rnn.BasicLSTMCell(
            #       size, forget_bias=0.0, state_is_tuple=True,
            #       reuse=tf.get_variable_scope().reuse)
            # else:
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.device("/cpu:0"):

            self.inputs = tf.placeholder(dtype=data_type(), shape=(batch_size, num_steps, embedding_size*3))
            self.targets = tf.placeholder(dtype=tf.int32, shape=(batch_size, num_steps))

        if is_training and config.keep_prob < 1:
            # Dropout allows to use the net for train and testing
            # See: https://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow
            # and: http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
            inputs = tf.nn.dropout(self.inputs, config.keep_prob)
        else:
            inputs = self.inputs

        inputs = tf.unstack(inputs, num=num_steps, axis=1)

        outputs, state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 126930
    embedding_size = 200
    epoch_size = 1

class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 35
    hidden_size = 512
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 126930
    embedding_size = 200
    epoch_size = 1

class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 1
    num_steps = 35
    hidden_size = 1024
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 126930
    embedding_size = 1000
    epoch_size = 1

class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 10
    vocab_size = 126930
    embedding_size = 200
    epoch_size = 1


def run_epoch(session, generator, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    config = model.config
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    print("Epoch size starting training %s" % config.epoch_size)
    sys.stdout.flush()
    for step in range(config.epoch_size):
        x, y = next(generator)
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        # feed_dict["embeddings"] = embeddings
        feed_dict[model.inputs] = x
        feed_dict[model.targets] = y

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += config.num_steps

        if verbose and step % 100 == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / config.epoch_size, np.exp(costs / iters),
                   iters * config.batch_size / (time.time() - start_time)))
            sys.stdout.flush()

    return np.exp(costs / iters)


def get_config():
    if FLAGS.model == "small":
        return SmallConfig()
    elif FLAGS.model == "medium":
        return MediumConfig()
    elif FLAGS.model == "large":
        return LargeConfig()
    elif FLAGS.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)

def get_epoch_size(files, config):
    total = 0
    for file in files:
        file_words = subprocess.check_output(['wc', '-w', file])
        number = file_words.split()[0]
        words = int(number)
        total += words - (words % (config.batch_size * config.num_steps))
    print("Total words: %s, Batch size: %s, Num steps: %s" % (total, config.batch_size, config.num_steps))
    sys.stdout.flush()
    epoch_size = ((total // config.batch_size) - 1) // config.num_steps

    return epoch_size

def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to wiki data directory list")

    vocab_size = 126930

    config = get_config()
    config.vocab_size = vocab_size

    valid_config = get_config()
    config.vocab_size = vocab_size


    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    eval_config.vocab_size = vocab_size

    embeddings = VectorManager.read_vector("%s%s.pklz" % (FLAGS.embeddings, config.embedding_size))

    # Load LDA or LSI model for topic creator
    if "lda" in FLAGS.context:
        model = LdaModel.load(FLAGS.topic_model_path)
    elif "lsi" in FLAGS.context:
        model = LsiModel.load(FLAGS.topic_model_path)
    else:
        model = None

    topic_creator = TopicCreator(FLAGS.dictionary_path, "%s%s.pklz" % (FLAGS.word2id_path, config.embedding_size),
                                 embeddings, model)
    files = open(FLAGS.data_path).read().split()

    training_list = files[0:int(0.8 * len(files))]
    validation_list = files[int(0.8 * len(files)):int(0.9 * len(files))]
    testing_list = files[int(0.9 * len(files)):len(files)]

    print("Lists sizes\n * Training: %s\n * Validation: %s\n * Testing: %s" %
          (len(training_list), len(validation_list), len(testing_list)))

    config.epoch_size = get_epoch_size(training_list, config)
    valid_config.epoch_size = get_epoch_size(validation_list, valid_config)
    eval_config.epoch_size = get_epoch_size(testing_list, eval_config)

    gen_train = generate_arrays_from_list("Train", topic_creator, training_list, embeddings, batch_size=config.batch_size,
                                          embedding_size=config.embedding_size, num_steps=config.num_steps)

    gen_valid = generate_arrays_from_list("Validation", topic_creator, validation_list, embeddings, batch_size=valid_config.batch_size,
                                          embedding_size=valid_config.embedding_size, num_steps=valid_config.num_steps)

    gen_test = generate_arrays_from_list("Test", topic_creator, testing_list, embeddings, batch_size=eval_config.batch_size,
                                         embedding_size=eval_config.embedding_size, num_steps=eval_config.num_steps)

    print("Epoch sizes\n * Training: %s\n * Validation: %s\n * Testing: %s" %
          (config.epoch_size, valid_config.epoch_size, eval_config.epoch_size))
    sys.stdout.flush()
    with tf.Graph().as_default():
        # Args: [minval, maxval]
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = WPModel(is_training=True, config=config)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Learning Rate", m.lr)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = WPModel(is_training=False, config=valid_config)
            tf.summary.scalar("Validation Loss", mvalid.cost)

        with tf.name_scope("Test"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = WPModel(is_training=False, config=eval_config)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, generator=gen_train, model=m, eval_op=m.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, generator=gen_valid, model=mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, generator=gen_test, model=mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
