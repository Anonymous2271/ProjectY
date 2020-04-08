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
import math
from y_layer import LinkMemoryLayer


class YModel(keras.layers.Layer):
    def __init__(self, n_layers, n_classes, width_layer, strides):
        super(YModel, self).__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.width_layer = width_layer
        self.strides = strides
        self.width_net = n_layers * width_layer

        # 低级特征采集层
        self.spec_conv = keras.layers.Conv2D(filters=8, kernel_size=[5, 5], strides=[2, 2], padding='valid',
                                             activation=tf.nn.leaky_relu, use_bias=True, data_format='channels_first')
        self.spec_pool = keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
                                                data_format='channels_first')
        # 链式记忆层
        self.lay1 = LinkMemoryLayer(filter=[8, 3, 3, 1, 1], myname='lay1')
        self.lay2 = LinkMemoryLayer(filter=[8, 3, 3, 1, 1], myname='lay2')
        self.lay3 = LinkMemoryLayer(filter=[8, 3, 3, 1, 1], myname='lay3')
        self.lay4 = LinkMemoryLayer(filter=[8, 3, 3, 1, 1], myname='lay4')
        self.lay5 = LinkMemoryLayer(filter=[4, 3, 3, 1, 1], myname='lay5')
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

        # h = math.floor(((input_shape[2] - self.filter1[1]) / self.filter1[3] + 1) / 2)
        # w = math.floor(((input_shape[3] - self.filter[2]) / self.filter[4] + 1) / 2)

        while index <= len_input - self.width_net:
            s1 = tf.slice(inputs, [0, 0, 0, index], [-1, -1, -1, self.width_layer])
            s2 = tf.slice(inputs, [0, 0, 0, index + self.width_layer], [-1, -1, -1, self.width_layer])
            s3 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 2], [-1, -1, -1, self.width_layer])
            s4 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 3], [-1, -1, -1, self.width_layer])
            s5 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 4], [-1, -1, -1, self.width_layer])

            s_fea1 = self.spec_pool(self.spec_conv(s1))
            s_fea2 = self.spec_pool(self.spec_conv(s2))
            s_fea3 = self.spec_pool(self.spec_conv(s3))
            s_fea4 = self.spec_pool(self.spec_conv(s4))
            s_fea5 = self.spec_pool(self.spec_conv(s5))

            f1 = self.lay1(inputs=s_fea1)
            f2 = self.lay2(inputs=tf.concat([s_fea2, f1], axis=1))
            f3 = self.lay3(inputs=tf.concat([s_fea3, f2], axis=1))
            f4 = self.lay4(inputs=tf.concat([s_fea4, f3], axis=1))
            f5 = self.lay5(inputs=tf.concat([s_fea5, f4], axis=1))
            # [?, 4, 38, 3]

            logits.append(f5)
            index += self.strides

        # print(logits[-1].get_shape().as_list())
        return self.gap(logits[-1])
