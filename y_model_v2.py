# coding: utf-8
# ---
# @File: y_model_v2.py
# @Time: 2020/8/10 14:50
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @desc: 
# ---

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from PIL import Image
import os
from y_v2 import FreqAttentionU

# 用来存储隐藏层特征图的子文件夹
h_layer_fea_dir = '5 层'


class YModel(keras.layers.Layer):
    def __init__(self, n_layers, n_classes, width_layer, strides):
        super(YModel, self).__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.width_layer = width_layer
        self.strides = strides
        self.width_net = (n_layers + 1) * width_layer
        self.i = 0  # 画图的batch计数器

        # 链式记忆层，参数不共享
        self.lay1 = FreqAttentionU(filter_para=[16, 3, 3, 3, 3], my_name='lay1', is_first_layer=True)
        self.lay2 = FreqAttentionU(filter_para=[16, 3, 3, 3, 3], my_name='lay2')
        self.lay3 = FreqAttentionU(filter_para=[16, 3, 3, 3, 3], my_name='lay3')
        self.lay4 = FreqAttentionU(filter_para=[16, 3, 3, 3, 3], my_name='lay4')
        self.lay5 = FreqAttentionU(filter_para=[8, 3, 3, 3, 3], my_name='lay5')

        # 分类层
        self.gap = tf.keras.layers.GlobalAvgPool2D(data_format='channels_first')
        self.frequency_matrix = self.add_weight(shape=[1, n_classes, 18, 5],
                                                initializer=keras.initializers.orthogonal, trainable=True)
        # self.flatten = keras.layers.Flatten(data_format='channels_first')
        # self.dense = tf.keras.layers.Dense(units=n_classes, activation=tf.nn.leaky_relu)

        # for i in range(n_layers):
        #     locals()['layer_' + str(i)] = link_memory_unit(width=width, n_filters1=8, n_filters2=4)

    def call(self, inputs, is_train=True, **kwargs):
        """
        :param is_train:
        :param inputs: [?, 1, 80, 600]
        :return: logits
        """
        # print('inputs: ', inputs.get_shape().as_list())

        len_input = inputs.get_shape().as_list()[-1]
        index = 0
        logits = []

        # h = math.floor(((input_shape[2] - self.filter1[1]) / self.filter1[3] + 1) / 2)
        # w = math.floor(((input_shape[3] - self.filter[2]) / self.filter[4] + 1) / 2)

        while index + self.width_net <= len_input:
            s1 = tf.slice(inputs, [0, 0, 0, index], [-1, -1, -1, self.width_layer])
            s2 = tf.slice(inputs, [0, 0, 0, index + self.width_layer], [-1, -1, -1, self.width_layer])
            s3 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 2], [-1, -1, -1, self.width_layer])
            s4 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 3], [-1, -1, -1, self.width_layer])
            s5 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 4], [-1, -1, -1, self.width_layer])
            # print('s1: ', s1.get_shape().as_list())
            # [?, ?, 18, 5]

            h1 = self.lay1(inputs=s1, hidden=s1)
            h2 = self.lay2(inputs=s2, hidden=h1)
            h3 = self.lay3(inputs=s3, hidden=h2)
            h4 = self.lay4(inputs=s4, hidden=h3)
            h5 = self.lay5(inputs=s5, hidden=h4)
            # f1 = self.lay1(inputs=tf.concat(s1, axis=1))
            # f2 = self.lay2(inputs=tf.concat([f1, s2], axis=1))
            # f3 = self.lay3(inputs=tf.concat([f2, s3], axis=1))
            # f4 = self.lay4(inputs=tf.concat([f3, s4], axis=1))
            # f5 = self.lay5(inputs=tf.concat([f4, s5], axis=1))
            # [?, ?, 18, 5]

            # 此次是最后一个时间步
            # if not is_train and index + self.width_net + self.strides > len_input:
            #     self.draw_hid_features(inputs=inputs, h_fea=tf.concat([f1, f2, f3, f4, f5], axis=-1))
            if index + self.width_net + self.strides > len_input:
                logits.append(h5)

            index += self.strides

        # print('logits[-1]', logits)
        class_weight = tf.tile(self.frequency_matrix, [64, 1, 1, 1], name="u_tiled")  # [bs, h, w, c]
        return self.gap(tf.multiply(logits[-1], class_weight))
        # return self.gap(logits[-1])
        # return self.dense(self.flatten(logits[-1]))
