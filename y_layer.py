# coding: utf-8
# ---
# @File: y_layer.py
# @description: 
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 4月04, 2020
# ---

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import math


class LinkMemoryLayer(keras.layers.Layer):
    def __init__(self, filter1=None, filter2=None, myname=None):
        """
        :param filter1:
        :param filter2:
        """
        super(LinkMemoryLayer, self).__init__()
        if filter1 is None:
            filter1 = [8, 5, 5, 2, 2]
        if filter2 is None:
            filter2 = [8, 3, 3, 1, 1]
        self.filter1 = filter1
        self.filter2 = filter2
        self.myname = myname

    def build(self, input_shape):
        # input_shape 是指原始的局部输入 [?, 1, 80, 20], 求出经过第一次卷积后的尺寸大小
        h = math.floor(((input_shape[2] - self.filter1[1]) / self.filter1[3] + 1) / 2)
        w = math.floor(((input_shape[3] - self.filter1[2]) / self.filter1[4] + 1) / 2)
        self.batch_size = input_shape[0]
        # print('h:', h, ' w:', w)
        self.Memory = self.add_weight(shape=[self.batch_size, self.filter2[0], h, w],
                                      initializer=keras.initializers.Ones(), trainable=False)
        self.U_weight = self.add_weight(shape=[1, self.filter2[0], h, h],
                                        initializer=keras.initializers.RandomNormal())
        self.V_weight = self.add_weight(shape=[1, self.filter2[0], h, h],
                                        initializer=keras.initializers.RandomNormal())

        self.conv1 = keras.layers.Conv2D(filters=self.filter1[0], kernel_size=[self.filter1[1], self.filter1[2]],
                                         strides=[self.filter1[3], self.filter1[4]], padding='valid',
                                         activation=tf.nn.leaky_relu, data_format='channels_first')
        self.conv2 = keras.layers.Conv2D(filters=self.filter2[0], kernel_size=[self.filter2[1], self.filter2[2]],
                                         strides=[self.filter2[3], self.filter2[4]], padding='same',
                                         activation=tf.nn.leaky_relu, data_format='channels_first')

        self.pooling1 = keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
                                               data_format='channels_first')

    def call(self, inputs,  in_forward):
        # in_spec = inputs[0]
        # in_forward = args

        feat_conv = self.conv1(inputs)
        feat_pool = self.pooling1(feat_conv)
        # print(feat_pool.get_shape().as_list(), 'sssss')

        # 在通道上进行拼接
        if in_forward is None:
            feat_relation = self.conv2(feat_pool)
        else:
            feat_relation = self.conv2(tf.concat([feat_pool, in_forward], axis=1))

        u_tiled = tf.tile(self.U_weight, [self.batch_size, 1, 1, 1], name="u_tiled")  # [bs, c, h, w]
        v_tiled = tf.tile(self.V_weight, [self.batch_size, 1, 1, 1], name="v_tiled")  # [bs, c, h, w]

        gate = tf.nn.tanh(tf.matmul(u_tiled, feat_relation)
                          + tf.matmul(v_tiled, self.Memory))
        self.Memory = tf.multiply(gate, self.Memory) + tf.multiply(1-gate, feat_relation)

        return self.Memory
