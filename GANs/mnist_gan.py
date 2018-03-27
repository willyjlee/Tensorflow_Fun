import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

batch_size = 50
noise_size = 50
num_iter = 1000000

class generator:
    def __init__(self):
        self.l1_size = 100
        self.w1 = tf.Variable(tf.truncated_normal(shape=[noise_size, self.l1_size], stddev=np.sqrt(2.0 / (noise_size))))
        self.b1 = tf.zeros([self.l1_size], tf.float32)
        self.w2 = tf.Variable(tf.truncated_normal(shape=[self.l1_size, 784], stddev=np.sqrt(2.0 / (self.l1_size))))
        self.b2 = tf.zeros([784], tf.float32)

    def outputs(self, z_holder):
        l1 = tf.nn.relu(tf.matmul(z_holder, self.w1) + self.b1)
        l2 = tf.nn.sigmoid(tf.matmul(l1, self.w2) + self.b2)
        return l2

class discriminator:
    def __init__(self):
        self.l1_size = 200
        self.w1 = tf.Variable(tf.truncated_normal(shape=[784, self.l1_size], stddev=np.sqrt(2.0 / (784))))
        self.b1 = tf.zeros([self.l1_size], tf.float32)
        self.w2 = tf.Variable(tf.truncated_normal(shape=[self.l1_size, 1], stddev=np.sqrt(2.0 / self.l1_size)))
        self.b2 = tf.zeros([1], tf.float32)

    def outputs(self, x_holder):
        l1 = tf.nn.relu(tf.matmul(x_holder, self.w1) + self.b1)
        l2 = tf.nn.sigmoid(tf.matmul(l1, self.w2) + self.b2)
        return l2

def setup():
    z_holder = tf.placeholder(tf.float32, shape=[None, noise_size])
    x_holder = tf.placeholder(tf.float32, shape=[None, 784])

    gen, dis = generator(), discriminator()
    gen_out, dis_out = gen.outputs(z_holder), dis.outputs(x_holder)
    dis_gen_out = dis.outputs(gen_out)
    dis_loss = -tf.reduce_mean(tf.log(dis_out) + tf.log(1.0 - dis_gen_out))
    gen_loss = -tf.reduce_mean(tf.log(dis_gen_out))

    dis_opt = tf.train.AdamOptimizer(1e-4).minimize(dis_loss)
    gen_opt = tf.train.AdamOptimizer(1e-4).minimize(gen_loss)
    return (dis_opt, dis_loss, gen_opt, gen_loss, gen_out), z_holder, x_holder

def disp(img):
    plt.imshow(img.reshape(28, 28), cmap='Greys_r')
    plt.show()

def train_loop():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    runs, z_holder, x_holder = setup()
    print('model done')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iter):
            x_batch, _ = mnist.train.next_batch(batch_size)
            z_rand = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_size])
            _, dis_loss = sess.run(runs[:2], feed_dict={z_holder: z_rand, x_holder: x_batch})

            z_rand = np.random.uniform(-1.0, 1.0, size=[batch_size, noise_size])
            _, gen_loss, gen_out = sess.run(runs[2:], feed_dict={z_holder: z_rand})

            if i % 2500 == 0:
                print("Iter: {}".format(i))
                print("disloss: {}".format(dis_loss))
                print("genloss: {}".format(gen_loss))
                disp(gen_out[np.random.randint(0, batch_size)])

train_loop()
