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
# print(my_block.trainable_weights)


y = np.array([2, 6, 4, 1, 8, 12, 23, 15, 6, 13])
x = range(len(y))
#
# 求点的拟合曲线
# parameter = np.polyfit(x=x, y=y, deg=4)  # 输出三次方程的参数
# p = np.poly1d(parameter)  # 根据参数输出方程
# print(p)
#
# plt.plot(p(x))
# plt.show()

y = np.array([2, 6, 4, 1, 8, 12, 23, 15, 6, 13])


def smooth(values, weight=0.5):
    """
    :param values: 点集合
    :param weight: 平滑度
    :return:
    """
    last = values[0]  # 上一个值
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point  # 这个方程是从 Tensorboard 源码中扒出来的
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


plt.plot(smooth(values=y))
plt.show()
