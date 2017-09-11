from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

batch_size = 10
patch_size = 5
depth = 40
num_hidden = 500
image_size = 28
num_channels = 1
num_labels = 10

def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

num_steps = 10000

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    graph = tf.Graph()

    with graph.as_default():
        test_dataset = tf.constant(np.reshape(mnist.test.images, newshape=[-1, image_size, image_size, 1]))
        train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        # conv1
        stddev1 = np.sqrt(2.0 / (patch_size * patch_size * num_channels))
        layer1_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=stddev1
        ))
        layer1_biases = tf.Variable(tf.zeros([depth]))

        # conv2
        stddev2 = np.sqrt(2.0 / (patch_size * patch_size * depth))
        layer2_weights = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=stddev2
        ))
        layer2_biases = tf.Variable(tf.zeros([depth]))

        # l3
        stretch_size = image_size // 4 * image_size // 4 * depth
        stddev3 = np.sqrt(2.0 / stretch_size)
        layer3_weights = tf.Variable(tf.truncated_normal(
            [stretch_size, num_hidden], stddev=stddev3
        ))
        layer3_biases = tf.Variable(tf.zeros(
            [num_hidden]
        ))

        # l4
        stddev4 = np.sqrt(2.0 / num_hidden)
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hidden, num_labels], stddev=stddev4
        ))
        layer4_biases = tf.Variable(tf.zeros([num_labels]))

        def model(data):
            layer1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(layer1 + layer1_biases)
            pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            layer2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(layer2 + layer2_biases)
            pool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            shape = pool2.get_shape().as_list()
            reshape = tf.reshape(pool2, [shape[0], -1])
            layer3 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

            layer4 = tf.matmul(layer3, layer4_weights) + layer4_biases
            return layer4

        logits = model(train_dataset)
        train_prediction = tf.nn.softmax(logits)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits)
        )

        #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        test_prediction = tf.nn.softmax(model(test_dataset))
    print('Graph all setup...')

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        for step in range(num_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = np.reshape(batch_x, newshape=[batch_size, image_size, image_size, 1])
            feed_dict = {train_dataset : batch_x, train_labels : batch_y}
            _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            # add validation?

            if step % 50 == 0:
                print('batch', step, 'loss:', l)
                acc = accuracy(predictions, batch_y)
                print('accuracy:', acc)
        test_accuracy = accuracy(test_prediction.eval(), mnist.test.labels)
        print('test accuracy:', test_accuracy)

if __name__ == '__main__':
    main()