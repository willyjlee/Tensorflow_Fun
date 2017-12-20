import numpy as np
import tensorflow as tf
import os
import random

# trying out bidirectional rnn
lstm_units = 30
input_max_len = 8
batch_size = 10
state_size = 30

input_size = 1
output_size = 2 * lstm_units # fwd & bwd
target_size = input_size

vocab_size = 26

num_iter = 20000
def build():

    tf.reset_default_graph()
    # forward cell
    f_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_units, state_is_tuple=True)
    # f_state = tf.placeholder(tf.float32, shape=[batch_size, state_size])
    # f_hidden_state = tf.placeholder(tf.float32, shape=[batch_size, state_size])

    # backward cell
    b_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_units, state_is_tuple=True)
    # b_state = tf.placeholder(tf.float32, shape=[batch_size, state_size])
    # b_hidden_state = tf.placeholder(tf.float32, shape=[batch_size, state_size])

    inputs = tf.placeholder(tf.float32, shape=(batch_size, input_max_len, 1))
    targets = tf.placeholder(tf.int32, shape=(batch_size, input_max_len))

    # bi rnn
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, inputs, dtype=tf.float32)
    # concat fwd & bwd
    concats = tf.concat(outputs, 2)

    stddev = np.sqrt(2.0 / (batch_size * output_size * vocab_size))
    w = tf.Variable(tf.truncated_normal(shape=[batch_size, output_size, vocab_size], stddev=stddev))
    stddev2 = np.sqrt(2.0 / (input_max_len * vocab_size))
    b = tf.Variable(tf.truncated_normal(shape=[input_max_len, vocab_size], stddev=stddev2))

    logits = tf.matmul(concats, w) + b   # [batch_size, input_max_len, vocab_size]
    predict = tf.argmax(logits, axis=2)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    opt = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # run
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = dict()
    for i in range(num_iter):
        feed_dict[inputs] = np.random.randint(0, 26, (batch_size, input_max_len, 1))
        # same lol
        feed_dict[targets] = np.reshape(feed_dict[inputs], (batch_size, input_max_len))
        l, optimizer, preds, inps, targs = sess.run([loss, opt, predict, inputs, targets], feed_dict=feed_dict)
        if i % 500 == 0:
            rand_ind = random.randint(0, batch_size - 1)
            print('Accuracy: {}'.format(l))
            #print('Input: {}'.format(feed_dict[inputs][rand_ind]))
            print('Target: {}'.format(feed_dict[targets][rand_ind]))
            print('Pred__: {}'.format(preds[rand_ind]))

build()