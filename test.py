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


def fun1(a):
    if a.shape == (1, 2):
        print("ok")
    else:
        raise Exception("shape error")

    return a * 2


var1 = np.random.randint(10, size=(2, 1, 2))
var2 = np.random.randint(10, size=(1, 2))

print(var1)
print(var1 > 5)

# fun1(var1) 执行错误 shape error
# fun1(var2) 可以执行

# 执行
rtn = tf.map_fn(fun1, var1)

# 结果打印
print(rtn)


