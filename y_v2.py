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
    def __init__(self, filter_para=None, my_name=None, is_first_layer=False):
        """
        :param filter_para:
        :param my_name:
        """
        super(FreqAttentionU, self).__init__()
        if filter_para is None:
            filter_para = [8, 3, 3, 1, 1]
        self.f_para = filter_para
        self.my_name = my_name
        self.is_first_layer = is_first_layer

    def build(self, input_shape):
        # [?, filter[0]*2, h, w]
        c = input_shape[1]
        h = input_shape[2]
        # w = int(input_shape[3]/2)
        w = input_shape[3]

        self.batch_size = input_shape[0]  # 这里的input_shape只取了第一次的，当最后一个批次的size有变化时，可能出错。

        self.flatten = k.layers.Flatten(data_format='channels_first')
        self.attention_freq = k.layers.Dense(units=40, activation=tf.nn.sigmoid, use_bias=False,
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

    def boolean_mask(self, a):
        median = np.median(a)
        return tf.less(median, a)

    def attention_pooling(self, hidden, inputs):
        batch_size = inputs.get_shape()[0]

        hidden = self.flatten(hidden)
        line = self.attention_freq(hidden)  # [?, units]
        mask_line = tf.map_fn(fn=self.boolean_mask, elems=line, dtype=bool)  # [?, units]

        print('hidden', hidden.get_shape())
        print('line', line.get_shape())
        print('mask_line', mask_line.get_shape())
        print('inputs', inputs.get_shape())

        # [?, h]
        # mask = tf.expand_dims(mask_line, axis=-1)  # [?, h, 1]
        # mask = tf.tile(mask, [1, 1, 20])  # [?, h, 20]
        # mask = tf.expand_dims(mask_line, axis=1)  # [?, 1, h, 20]
        # mask = tf.tile(mask, [1, c, 1])  # [?, 8, h, 20]
        # print('mask', mask.get_shape())

        feat_attention = tf.boolean_mask(tensor=inputs, mask=mask_line, axis=0)
        feat_attention = tf.split(feat_attention, num_or_size_splits=batch_size, axis=0)
        feat_attention = tf.cast(feat_attention, dtype='int32')

        return feat_attention

    def call(self, inputs, hidden=None, **kwargs):
        # [?, filter[0], h, w]
        if self.is_first_layer:
            inputs = k.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], padding='valid', data_format='channels_first')(inputs)

        # inputs = tf.concat(inputs, axis=1)
        new_b = self.attention_pooling(hidden, inputs)
        # print('new_b', new_b.get_shape())

        relation = tf.concat([hidden, new_b], axis=1)

        print('relation', relation.get_shape())
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
