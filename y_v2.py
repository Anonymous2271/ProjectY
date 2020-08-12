# coding: utf-8
# ---
# @File: y_v2.py
# @Time: 2020/8/6 15:35
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @desc: 
# ---

import numpy as np
import tensorflow as tf
import tensorflow.keras as k


class FreqAttentionU(k.layers.Layer):
    def __init__(self, filter_para=None, my_name=None, is_last_layer=False):
        """
        :param filter_para:
        :param my_name:
        """
        super(FreqAttentionU, self).__init__()
        if filter_para is None:
            filter_para = [8, 3, 3, 1, 1]
        self.f_para = filter_para
        self.my_name = my_name
        self.is_last_layer = is_last_layer

    def build(self, input_shape):
        # [2, ?, filter[0]*2, h, w]
        c = input_shape[2]
        h = input_shape[3]
        # w = int(input_shape[3]/2)
        w = input_shape[4]

        self.batch_size = input_shape[0]  # 这里的input_shape只取了第一次的，当最后一个批次的size有变化时，可能出错。

        self.attention_freq = k.layers.Dense(units=h, activation=tf.nn.sigmoid, use_bias=False,
                                             kernel_initializer='glorot_uniform')

        self.conv_relation = k.layers.Conv2D(filters=self.f_para[0],
                                             kernel_size=[self.f_para[1], self.f_para[2]],
                                             strides=[1, 1], padding='same',
                                             activation=None, data_format='channels_first',
                                             kernel_initializer=k.initializers.glorot_uniform,
                                             use_bias=False)
        # self.pool_relation = keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
        #                                             data_format='channels_first')
        self.batch_norm = k.layers.BatchNormalization()
        self.activate = k.layers.LeakyReLU()

        self.Memory = self.add_weight(shape=[self.batch_size, self.f_para[0], h, w],
                                      initializer=k.initializers.orthogonal, trainable=False)
        # 版本 4  的参数 ####################
        self.conv_gate = k.layers.Conv2D(filters=self.f_para[0], kernel_size=[self.f_para[3], self.f_para[4]],
                                         strides=[1, 1], padding='same',
                                         activation=tf.nn.sigmoid, data_format='channels_first',
                                         kernel_initializer=k.initializers.glorot_uniform,
                                         bias_initializer=k.initializers.zeros)

    def bool_mask(self, a):
        median = np.median(a)
        return a > median

    def attention_pooling(self, feat_a, feat_b):
        line = self.attention_freq(feat_a).numpy()
        mask_line = tf.map_fn(fn=self.bool_mask, elems=line)

        # [?, h]
        mask = tf.expand_dims(mask_line, axis=-1)  # [?, h, 1]
        mask = tf.tile(mask, [1, 1, 20])  # [?, h, 20]
        mask = tf.expand_dims(mask, axis=1)  # [?, 1, h, 20]
        mask = tf.tile(mask, [1, 8, 1, 1])  # [?, 8, h, 20]

        feat_attention = tf.boolean_mask(tensor=feat_b, mask=mask)

        return feat_attention

    def call(self, inputs, **kwargs):
        # [?, filter[0], h, w]
        feat_a = inputs[0]
        feat_b = inputs[1]
        # inputs = tf.concat(inputs, axis=1)
        new_b = self.attention_pooling(feat_a, feat_b)

        relation = tf.concat([feat_a, new_b], axis=1)

        feat_relation = self.conv_relation(relation)
        feat_relation = self.batch_norm(feat_relation)
        feat_relation = self.activate(feat_relation)

        # 版本 4 ：使用卷积
        ################################
        gate = self.conv_gate(tf.concat([feat_relation, self.Memory], axis=1))

        # 门控
        self.Memory = tf.multiply(gate, self.Memory) + tf.multiply(1 - gate, feat_relation)
        self.Memory = tf.nn.tanh(self.Memory)
        return self.Memory
