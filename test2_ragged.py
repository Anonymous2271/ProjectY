# coding: utf-8
# ---
# @File: test2_ragged.py
# @Time: 2020/8/14 15:10
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @desc: 
# ---
import numpy as np
import tensorflow as tf

a = np.random.randint(10, size=[2, 2, 4, 4])
line = np.random.randint(10, size=[2, 4])


def to_bool(x):
    median = np.median(x)
    return tf.less(median, x)


mask_line = tf.map_fn(fn=to_bool, elems=line, dtype=bool)  # [?, units]
# [?, h]
# mask = tf.expand_dims(mask_line, axis=-1)
# mask = tf.tile(mask, [1, 1, 4])
mask = tf.expand_dims(mask_line, axis=1)
mask = tf.tile(mask, [1, 2, 1])
print('mask', mask.get_shape())

feat_attention = tf.ragged.boolean_mask(data=a, mask=mask)
feat_attention.to_tensor()
print('a', a)
print('mask', mask)
print('feat_attention', feat_attention)

