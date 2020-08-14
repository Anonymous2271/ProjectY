# coding: utf-8
# ---
# @File: test3_top.py
# @Time: 2020/8/14 15:51
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @desc: 
# ---
import numpy as np
import tensorflow as tf

a = np.random.randint(10, size=[2, 4, 4, 2])
line = np.random.randint(10, size=[2, 4])

attention = tf.nn.top_k(line, 2, sorted=False).indices
attention = tf.sort(attention, direction='ASCENDING')

print('line', line)
print('attention', attention)

out = tf.sparse.mask(a, mask_indices=attention)

print(out)
