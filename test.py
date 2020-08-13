# coding: utf-8
# ---
# @File: test.py
# @description:
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 3月22, 2020
# ---

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import make_lsq_spline, BSpline

# class MyLayer(tf.keras.layers.Layer):
#     def __init__(self, unit=32):
#         super(MyLayer, self).__init__()
#         self.unit = unit
#
#     def build(self, input_shape):
#         # self.weight = self.add_weight(shape=(input_shape[-1], self.unit),
#         #                               initializer=tf.keras.initializers.RandomNormal(),
#         #                               trainable=True)
#         # self.bias = self.add_weight(shape=(self.unit,),
#         #                             initializer=tf.keras.initializers.Zeros(),
#         #                             trainable=True)
#         self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=[3, 3],
#                                             strides=[1, 1], padding='valid',
#                                             activation=tf.nn.leaky_relu, data_format='channels_first', trainable=True)
#         self.dense = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
#
#     def call(self, inputs):
#         # a = tf.matmul(inputs, self.weight) + self.bias
#         a = self.conv1(inputs)
#
#         return self.dense(a)
#
#
# class MyBlock(tf.keras.layers.Layer):
#     def __init__(self):
#         super(MyBlock, self).__init__()
#         self.layer1 = MyLayer(32)
#         self.layer2 = MyLayer(16)
#         self.layer3 = MyLayer(2)
#
#     def call(self, inputs):
#         h1 = self.layer1(inputs)
#         h1 = tf.nn.relu(h1)
#         h2 = self.layer2(h1)
#         h2 = tf.nn.relu(h2)
#         return self.layer3(h2)
#
#
# my_block = MyBlock()
# print('trainable weights:', len(my_block.trainable_weights))
# y = my_block(tf.ones(shape=(10, 3, 64, 64)))
# # 构建网络在build()里面，所以执行了才有网络
# print('trainable weights:', len(my_block.trainable_weights))
# print(my_block.trainable_weights


def bool_mask(x):
    median = np.median(x)
    return tf.less(median, x)


a = np.random.randint(10, size=[2, 4, 4])
line = np.random.randint(10, size=[2, 4])

mask = tf.map_fn(fn=bool_mask, elems=line, dtype=bool)
# [2, 4]

# mask = tf.expand_dims(mask, axis=-1)  # [?, h, 1]
# mask = tf.tile(mask, [1, 1, 4])  # [?, h, 4]
# mask = tf.expand_dims(mask, axis=-1)  # [2, 4, 1]
# mask = tf.tile(mask, [1, 1, 1])  # [2, 4, 1]

feat_attention = tf.boolean_mask(a, mask=mask, axis=0)
feat_attention = tf.split(feat_attention, num_or_size_splits=2, axis=0)
feat_attention = tf.cast(feat_attention, dtype='int32')

# attention_pos = tf.nn.top_k(a, k=1)
# print(a)
# pool = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], padding='valid', data_format='channels_first')
# b = pool(a)
# print(b)
#
# attention = tf.nn.top_k(line, 2, sorted=False).indices
print('line', line)
print('mask', mask)
print('a', a)
print('att', feat_attention)
#
# i = 0
#
#
# def cond(i, a, att):
#     return tf.less(i, np.shape(a)[0])
#
#
# def body(i, a, att):
#     new = tf.gather(params=a[i], indices=att[i])
#     i = i+1
#     return new
#
#
# out = tf.while_loop(cond, body, [i, a, attention])
#
# # new = tf.gather_nd(params=a, indices=attention, batch_dims=0)
#
#
# print('a', a)
# print('new', out)
