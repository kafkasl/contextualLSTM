from utils.vector_manager import VectorManager
from utils.flatten import flatten
import tensorflow as tf
import numpy as np


data = ['consumers', 'may', 'want', 'to', 'move', 'their',
        'telephones', 'a', 'little', 'closer', 'to', 'the',
        'tv', 'set', '<unk>', '<unk>', 'watching', 'abc', "'s",
        'monday', 'night', 'football', 'can', 'now', 'vote', 'during',
        '<unk>', 'for', 'the', 'greatest', 'play', 'in', 'N', 'years',
        'from', 'among', 'four', 'or', 'five', '<unk>', '<unk>',
        'two', 'weeks', 'ago', 'viewers', 'of', 'several', 'nbc',
        '<unk>', 'consumer', 'segments', 'started', 'calling', 'a',
        'N', 'number', 'for', 'advice', 'on', 'various', '<unk>',
        'issues', 'and', 'the', 'new', 'syndicated', 'reality',
        'show', 'hard', 'copy', 'records', 'viewers', "'", 'opinions',
        'for', 'possible', 'airing', 'on', 'the', 'next', 'day', "'s",
        'show', 'interactive', 'telephone', 'technology', 'has',
        'taken', 'a', 'new', 'leap', 'in', '<unk>', 'and', 'television',
        'programmers', 'are', 'racing', 'to', 'exploit']



from lstm.reader_wp import wiki_raw_data, wiki_producer

train, valid, test = wiki_raw_data("../data/wikipedia/")
#data = data.flatten()
batch_size = 2
num_steps = 3
inputs, targets, s1, s2, x = wiki_producer(train, batch_size=batch_size, num_steps=num_steps)

# print inputs
# sv = tf.train.Supervisor()
# with sv.managed_session() as sess:
#     print sess.run([inputs, s1, s2])
#     print sess.run([inputs, s1, s2])
#     print sess.run([inputs, s1, s2])
#     print sess.run([inputs, s1, s2])
#     print sess.run([inputs, s1, s2])

data_len = np.size(train)
batch_len = data_len // batch_size
ndata = np.reshape(train[0: batch_size * batch_len],
                  [batch_size, batch_len])


