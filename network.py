import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt

from abc import *


class Network(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name):
        self.name = name
        self.initializer = tf.truncated_normal_initializer(stddev=0.01)

    @abstractmethod
    def build(self, input):
        pass


class Generator(Network):
    def __init__(self, name, n_input):
        super(Generator, self).__init__(name)

        self.n_input = n_input

    def build(self, input):
        with tf.variable_scope(self.name):
            deconv1_1 = slim.conv2d_transpose(inputs=input, num_outputs=64, kernel_size=[2, 2], padding='SAME',
                                              scope='deconv1_1', weights_initializer=self.initializer,
                                              normalizer_fn=slim.batch_norm, activation_fn=slim.relu)

            deconv2_1 = slim.conv2d_transpose(inputs=deconv1_1, num_outputs=128, kernel_size=[2, 2], padding='SAME',
                                              scope='deconv2_1', weights_initializer=self.initializer,
                                              normalizer_fn=slim.batch_norm, activation_fn=slim.relu)

            deconv3_1 = slim.conv2d_transpose(inputs=deconv2_1, num_outputs=256, kernel_size=[2, 2], padding='SAME',
                                              scope='deconv3_1', weights_initializer=self.initializer,
                                              normalizer_fn=slim.batch_norm, activation_fn=slim.relu)
            deconv3_2 = slim.conv2d_transpose(inputs=deconv3_1, num_outputs=256, kernel_size=[2, 2], padding='SAME',
                                              scope='deconv3_1', weights_initializer=self.initializer,
                                              normalizer_fn=slim.batch_norm, activation_fn=slim.relu)

            deconv4_1 = slim.conv2d_transpose(inputs=deconv3_2, num_outputs=512, kernel_size=[3, 3], padding='SAME',
                                              scope='deconv4_1', weights_initializer=self.initializer, stride=2,
                                              normalizer_fn=slim.batch_norm, activation_fn=slim.relu)
            deconv4_2 = slim.conv2d_transpose(inputs=deconv4_1, num_outputs=512, kernel_size=[3, 3], padding='SAME',
                                              scope='deconv4_2', weights_initializer=self.initializer, stride=2,
                                              normalizer_fn=slim.batch_norm, activation_fn=slim.relu)



class Discriminator(Network):
    def __init__(self, name):
        super(Discriminator, self).__init__(name)

    def build(self, input):
        with tf.variable_scope(self.name):
            conv1_1 = slim.conv2d(inputs=input, num_outputs=64, kernel_size=[2, 2], scope='conv1_1',
                                  weights_initializer=self.initializer,
                                  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)

            conv2_1 = slim.conv2d(inputs=conv1_1, num_outputs=128, kernel_size=[2, 2], scope='conv2_1',
                                  weights_initializer=self.initializer,
                                  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)

            conv3_1 = slim.conv2d(inputs=conv2_1, num_outputs=256, kernel_size=[2, 2], padding='SAME', scope='conv3_1',
                                  stride=2, weights_initializer=self.initializer,
                                  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
            conv3_2 = slim.conv2d(inputs=conv3_1, num_outputs=256, kernel_size=[2, 2], scope='conv3_2',
                                  stride=2, weights_initializer=self.initializer,
                                  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)

            conv4_1 = slim.conv2d(inputs=conv3_2, num_outputs=512, kernel_size=[3, 3], scope='conv4_1',
                                  stride=2, weights_initializer=self.initializer,
                                  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)
            conv4_2 = slim.conv2d(inputs=conv4_1, num_outputs=512, kernel_size=[3, 3], padding='SAME', scope='conv4_2',
                                  stride=2, weights_initializer=self.initializer,
                                  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.leaky_relu)

            # global average pooling
            # 그 다음에는 dense ??

