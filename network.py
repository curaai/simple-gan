import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


class Generator:
    def __init__(self, n_noise, n_hidden, n_input):
        self.n_noise = n_noise
        self.n_hidden = n_hidden
        self.n_input = n_input

        self.name = 'Generator'

    def generate(self, Z):
        with tf.variable_scope(self.name):
            G_W1 = tf.Variable(tf.random_normal([self.n_noise, self.n_hidden], stddev=0.01))
            G_W1 = tf.nn.relu(tf.matmul(Z, G_W1))

            G_W2 = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], stddev=0.01))
            G_W2 = tf.nn.relu(tf.matmul(G_W1, G_W2))

            G_W3 = tf.Variable(tf.random_normal([self.n_hidden, self.n_input], stddev=0.01))
            output = tf.nn.sigmoid(tf.matmul(G_W2, G_W3))

            return output


class Discriminator:
    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.name = 'Discriminator'

    def discriminate(self, image):
        with tf.variable_scope(self.name):
            D_W1 = tf.Variable(tf.random_normal([self.n_input, self.n_hidden], stddev=0.01))
            D_W1 = tf.nn.relu(tf.matmul(image, D_W1))

            D_W2 = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], stddev=0.01))
            D_W2 = tf.nn.relu(tf.matmul(D_W1, D_W2))

            D_W3 = tf.Variable(tf.random_normal([self.n_hidden, 1], stddev=0.01))
            output = tf.nn.sigmoid(tf.matmul(D_W2, D_W3))

            return output


def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))


def main():
    total_epoch = 200
    batch_size = 100
    learning_rate = 0.0002

    total_batch = int(mnist.train.num_examples / batch_size)

    n_hidden = 512
    n_noise = 128
    n_input = 28 * 28

    G = Generator(n_noise, n_hidden, n_input)
    D = Discriminator(n_input, n_hidden)

    X = tf.placeholder(tf.float32, [None, n_input])
    Z = tf.placeholder(tf.float32, [None, n_noise])

    g = G.generate(Z)

    D_fake = D.discriminate(g)
    D_real = D.discriminate(X)

    loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))
    loss_G = tf.reduce_mean(tf.log(D_fake))

    vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=D.name)
    vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G.name)

    train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=vars_D)
    train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=vars_G)

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(total_epoch):
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                noise = get_noise(batch_size, n_noise)

                _, loss_var_D = sess.run([train_D, loss_D], feed_dict={X: batch_x, Z:noise})
                _, loss_var_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

            print('Epoch:', '%04d' % epoch)

            #########
            # 학습이 되어가는 모습을 보기 위해 주기적으로 이미지를 생성하여 저장
            ######
            if epoch == 0 or (epoch + 1) % 10 == 0:
                sample_size = 10
                noise = get_noise(sample_size, n_noise)
                samples = sess.run(g, feed_dict={Z: noise})

                fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

                for i in range(sample_size):
                    ax[i].set_axis_off()
                    ax[i].imshow(np.reshape(samples[i], (28, 28)))

                plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
                plt.close(fig)


if __name__ == '__main__':
    main()
