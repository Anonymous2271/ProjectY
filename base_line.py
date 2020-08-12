# coding: utf-8
# ---
# @File: base_line.py
# @description: 对比实验
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 5月4, 2020
# ---

import tensorflow as tf
import tensorflow.keras as keras


class CNN_GRU(keras.Model):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = keras.layers.Conv2D(filters=16, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                         activation=None, use_bias=False, data_format='channels_first',
                                         kernel_initializer=keras.initializers.glorot_uniform)
        self.conv2 = keras.layers.Conv2D(filters=8, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                         activation=None, use_bias=False, data_format='channels_first',
                                         kernel_initializer=keras.initializers.glorot_uniform)
        self.activate = keras.layers.LeakyReLU()
        self.pool = keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
                                           data_format='channels_first')
        self.bn = keras.layers.BatchNormalization()
        self.gru = keras.layers.GRU(units=16)
        self.fla = keras.layers.Flatten(data_format='channels_first')
        self.den = keras.layers.Dense(units=self.n_classes)

    def call(self, inputs, is_train=True, mask=None):
        o1 = self.conv1(inputs)
        o1 = self.bn(o1)
        o1 = self.activate(o1)
        o1 = self.pool(o1)

        o2 = self.conv2(o1)
        o2 = self.bn(o2)
        o2 = self.activate(o2)
        o2 = self.pool(o2)
        # [bs, c, 20, 150]

        x_rnns = tf.unstack(o2, axis=1)  # 展开通道维度  c*[?, 20, 150]
        x_rnn = tf.concat(x_rnns, axis=1)  # 合并到列维度  [?, 20*c, 150]
        x_rnn = tf.transpose(x_rnn, perm=[0, 2, 1])  # [?, 150, 20*c]

        out = self.gru(x_rnn)

        return self.den(self.fla(out))



