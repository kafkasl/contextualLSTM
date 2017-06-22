# coding: utf-8
valid_data = open("simple-examples/data/ptb.valid.txt", "rb").read()
import tensorflow as tf
valid_data = [s.split() for s in valid_data]
valid_data[3]
dt = []
[dt.extend(s) for s in valid_data]
dt
valid_data
valid_data = open("simple-examples/data/ptb.valid.txt", "rb").read()
valid_data
valid_data = valid_data.split("\n")
[dt.extend(s) for s in valid_data]
dt
valid_data
valid_data = [s.split() for s in valid_data]
valid_data
[dt.extend(s) for s in valid_data]
dt
valid_data
[dt.extend(s) for s in valid_data]
dt
valid_data
dt = []
[dt.extend(s) for s in valid_data]
dt
dt
dat = dt[0:100]
get_ipython().magic(u'paste')
batch_size = 20
get_ipython().magic(u'paste')
num_steps = 10
get_ipython().magic(u'paste')
epoch_size
with tf.Session() as sess:
    print sess.run([epoch_size])
    
num_steps = 5
with tf.Session() as sess:
    print sess.run([epoch_size])
    
batch_size = 10
with tf.Session() as sess:
    print sess.run([epoch_size])
    
dat
len(dat)
with tf.Session() as sess:
    print sess.run([epoch_size])
    
with tf.Session() as sess:
    print sess.run([batch_len])
    
with tf.Session() as sess:
    print sess.run([batch_len, data])
    
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data])
    
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size])
    
num_steps
num_steps = 2
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size])
    
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data])
    
get_ipython().magic(u'paste')
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data])
    
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size])
    
get_ipython().magic(u'paste')
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size])
    
tf.reset_default_graph()
get_ipython().magic(u'paste')
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size])
    
get_ipython().magic(u'paste')
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size])
    
i = 0
get_ipython().magic(u'paste')
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size, x, y])
    
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size, x, y])
    
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size, x, y])
    
i = tf.train.range_input_producer(epoch_size, shuffle=False).
i = tf.train.range_input_producer(epoch_size, shuffle=False)
with tf.Session() as sess:
    print sess.run([data_len, batch_len, data, epoch_size, x, y, i])
    
get_ipython().magic(u'save')
get_ipython().magic(u'save reader_sess 1-69')
