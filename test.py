# coding: utf-8
# ---
# @File: test.py
# @description: RNN模拟
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 3月22, 2020
# ---

import numpy as np
import tensorflow as tf
from PIL import Image

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
# print(my_block.trainable_weights)


# the format of value: [NHWC]
shape = (28, 28)
initializer = tf.initializers.he_normal()
value = tf.Variable(initializer(shape=shape))

# conv_gate = tf.keras.layers.Conv2D(filters=3, kernel_size=[5, 5], strides=[1, 1],
#                                    padding='valid', activation=tf.nn.leaky_relu, data_format='channels_first')
#
# deconv_gate = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=[5, 5], strides=[1, 1],
#                                               padding='valid', activation=tf.nn.leaky_relu, data_format='channels_first')
#
# a = conv_gate(value)
#
# b = deconv_gate(a)
#
# print(np.shape(a))
# print(np.shape(b))

import tensorflow as tf
import cv2

# Create a 3x3 Gabor filter
params = {'ksize':(3, 3), 'sigma':1.0, 'theta': 0, 'lambd':15.0, 'gamma':0.02}
filter = cv2.getGaborKernel(**params)
# make the filter to have 4 dimensions.
filter = tf.expand_dims(filter, 2)
filter = tf.expand_dims(filter, 3)

# Apply the filter on `image`
answer = tf.conv2d(image, filter, strides=[1, 1, 1, 1], padding='SAME')