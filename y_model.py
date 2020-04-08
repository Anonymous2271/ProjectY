# coding: utf-8
# ---
# @File: y_model.py
# @description: 
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 4月05, 2020
# ---

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from y_layer import LinkMemoryLayer


class YModel(keras.layers.Layer):
    def __init__(self, n_layers, n_classes, width, strides):
        super(YModel, self).__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.width = width
        self.strides = strides
        self.width_net = n_layers * width

        # 低级特征采集层
        self.spec_layer = keras.layers.Conv2D(filters=8, kernel_size=[5, 5], strides=[2, 2], padding='valid',
                                              activation=tf.nn.leaky_relu, data_format='channels_first')
        # 链式记忆层
        self.lay1 = LinkMemoryLayer(filter1=[8, 5, 5, 2, 2], filter2=[8, 3, 3, 1, 1], myname='lay1')
        self.lay2 = LinkMemoryLayer(filter1=[8, 5, 5, 2, 2], filter2=[8, 3, 3, 1, 1], myname='lay2')
        self.lay3 = LinkMemoryLayer(filter1=[8, 5, 5, 2, 2], filter2=[8, 3, 3, 1, 1], myname='lay3')
        self.lay4 = LinkMemoryLayer(filter1=[8, 5, 5, 2, 2], filter2=[8, 3, 3, 1, 1], myname='lay4')
        self.lay5 = LinkMemoryLayer(filter1=[8, 5, 5, 2, 2], filter2=[4, 3, 3, 1, 1], myname='lay5')
        self.gap = tf.keras.layers.GlobalAvgPool2D(data_format='channels_first')

        # for i in range(n_layers):
        #     locals()['layer_' + str(i)] = link_memory_unit(width=width, n_filters1=8, n_filters2=4)

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: [?, 1, 80, 600]
        :return: logits
        """
        len_input = np.shape(inputs)[-1]
        index = 0
        logits = []
        while index <= len_input - self.width_net:
            fragment1 = tf.slice(inputs, [0, 0, 0, index], [-1, -1, -1, self.width])
            fragment2 = tf.slice(inputs, [0, 0, 0, index + self.width], [-1, -1, -1, self.width])
            fragment3 = tf.slice(inputs, [0, 0, 0, index + self.width * 2], [-1, -1, -1, self.width])
            fragment4 = tf.slice(inputs, [0, 0, 0, index + self.width * 3], [-1, -1, -1, self.width])
            fragment5 = tf.slice(inputs, [0, 0, 0, index + self.width * 4], [-1, -1, -1, self.width])

            f1 = self.lay1(inputs=fragment1, in_forward=None)
            f2 = self.lay2(inputs=fragment2, in_forward=f1)
            f3 = self.lay3(inputs=fragment3, in_forward=f2)
            f4 = self.lay4(inputs=fragment4, in_forward=f3)
            f5 = self.lay5(inputs=fragment5, in_forward=f4)
            # [?, 4, 38, 3]

            logits.append(f5)
            index += self.strides

        # print(logits[-1].get_shape().as_list())
        return self.gap(logits[-1])
