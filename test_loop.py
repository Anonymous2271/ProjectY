# coding: utf-8
# ---
# @File: test_loop.py
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


a = np.random.randint(10, size=[2, 4, 4, 2])
line = np.random.randint(10, size=[2, 4])


def to_bool(x):
    median = np.median(x)
    return tf.less(median, x)


# @tf.function
# def bool_mask(x, mask):
#     x = tf.concat(x, axis=-1)
#     return tf.boolean_mask(x, mask=mask, axis=0)
#
#
mask = tf.map_fn(fn=to_bool, elems=line, dtype=bool)
#
# feat_attention = bool_mask(x=a, mask=mask)
# [2, 4]


def cond(a, mask, i):
    batch_size = tf.shape(a)[0]
    return tf.less(i, batch_size)


def body(a, mask, i):
    item_a = a[i]
    item_mask = mask[i]
    item_mask = tf.concat(item_mask, axis=1)
    feat_att = tf.boolean_mask(item_a, mask=item_mask, axis=0)
    return feat_att


feat_attention = tf.while_loop(cond, body, [a, mask, 0])

# mask = tf.expand_dims(mask, axis=-1)  # [?, h, 1]
# mask = tf.tile(mask, [1, 1, 4])  # [?, h, 4]
# mask = tf.expand_dims(mask, axis=-1)  # [2, 4, 1]
# mask = tf.tile(mask, [1, 1, 1])  # [2, 4, 1]

# feat_attention = tf.boolean_mask(a, mask=mask, axis=2)

print('line', line)
print('mask', mask)
print('a', a)
print('att', feat_attention)

# feat_attention = tf.split(feat_attention, num_or_size_splits=2, axis=0)
# feat_attention = tf.cast(feat_attention, dtype='int32')

# attention_pos = tf.nn.top_k(a, k=1)
# print(a)
# pool = tf.keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], padding='valid', data_format='channels_first')
# b = pool(a)
# print(b)
#
# attention = tf.nn.top_k(line, 2, sorted=False).indices


#
#
# out = tf.while_loop(cond, body, [i, a, attention])
#
# # new = tf.gather_nd(params=a, indices=attention, batch_dims=0)
#
#
# print('a', a)
# print('new', out)
