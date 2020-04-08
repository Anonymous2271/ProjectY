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
    def __init__(self, filter=None, myname=None):
        """
        :param filter:
        :param myname:
        """
        super(LinkMemoryLayer, self).__init__()
        if filter is None:
            filter = [8, 3, 3, 1, 1]
        self.filter = filter
        self.myname = myname

    def build(self, input_shape):
        h = input_shape[2]
        w = input_shape[3]
        self.batch_size = input_shape[0]  # 这里的input_shape只取了第一次的，当最后一个批次的size有变化时，可能出错。

        self.conv = keras.layers.Conv2D(filters=self.filter[0], kernel_size=[self.filter[1], self.filter[2]],
                                        strides=[self.filter[3], self.filter[4]], padding='same',
                                        activation=tf.nn.leaky_relu, data_format='channels_first')

        # 版本 1 的参数 #################
        self.Memory = self.add_weight(shape=[self.batch_size, self.filter[0], h, w],
                                      initializer=keras.initializers.he_normal(), trainable=False)
        self.U_weight = self.add_weight(shape=[1, self.filter[0], h, h],
                                        initializer=keras.initializers.he_normal())
        self.V_weight = self.add_weight(shape=[1, self.filter[0], h, h],
                                        initializer=keras.initializers.he_normal())
        # 版本 2 的参数##################
        self.conv_gate = keras.layers.Conv2D(filters=self.filter[0], kernel_size=[self.filter[1], self.filter[2]],
                                             strides=[self.filter[3], self.filter[4]], padding='same',
                                             activation=tf.nn.leaky_relu, data_format='channels_first')

    def call(self, inputs, **kwargs):
        feat_conv = self.conv(inputs)
        # print(feat_conv.get_shape().as_list(), 'sssss')
        # batch_size = inputs.get_shape().as_list()[0]

        ################################
        # 版本 1 ：使用 GRU 的门控逻辑，优点是1)参数少，需要训练的参数就两个；2)可以做到与卷积通道相对应，最后应用GAP
        ################################
        u_tiled = tf.tile(self.U_weight, [self.batch_size, 1, 1, 1], name="u_tiled")  # [bs, c, h, w]
        v_tiled = tf.tile(self.V_weight, [self.batch_size, 1, 1, 1], name="v_tiled")  # [bs, c, h, w]

        gate = tf.nn.sigmoid(tf.matmul(u_tiled, feat_conv) + tf.matmul(v_tiled, self.Memory))
        out = tf.multiply(gate, self.Memory) + tf.multiply(1 - gate, feat_conv)
        self.Memory = tf.tanh(out)

        ################################
        # 版本 2 ：使用全连接层作为门控逻辑
        ################################

        return out
