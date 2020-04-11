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
from PIL import Image
import os
import time
from y_layer import LinkMemoryLayer


class YModel(keras.layers.Layer):
    def __init__(self, n_layers, n_classes, width_layer, strides):
        super(YModel, self).__init__()
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.width_layer = width_layer
        self.strides = strides
        self.width_net = (n_layers + 1) * width_layer
        self.i = 0  # 画图的batch计数器

        # 低级特征采集层，参数共享
        self.spec_conv = keras.layers.Conv2D(filters=8, kernel_size=[5, 5], strides=[2, 2], padding='valid',
                                             activation=tf.nn.leaky_relu, use_bias=True, data_format='channels_first')
        self.spec_pool = keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
                                                data_format='channels_first')
        # 链式记忆层，参数不共享
        self.lay1 = LinkMemoryLayer(filter_para=[8, 3, 3, 1, 1], my_name='lay1')
        self.lay2 = LinkMemoryLayer(filter_para=[8, 3, 3, 1, 1], my_name='lay2')
        self.lay3 = LinkMemoryLayer(filter_para=[8, 3, 3, 1, 1], my_name='lay3')
        self.lay4 = LinkMemoryLayer(filter_para=[8, 3, 3, 1, 1], my_name='lay4')
        # 降维分类
        # self.gap = tf.keras.layers.GlobalAvgPool2D(data_format='channels_first')
        self.flatten = keras.layers.Flatten(data_format='channels_first')
        self.dense = tf.keras.layers.Dense(units=4, activation=tf.nn.sigmoid)

        # for i in range(n_layers):
        #     locals()['layer_' + str(i)] = link_memory_unit(width=width, n_filters1=8, n_filters2=4)

    def call(self, inputs, is_train=True, **kwargs):
        """
        :param is_train:
        :param inputs: [?, 1, 80, 600]
        :return: logits
        """
        len_input = np.shape(inputs)[-1]
        index = 0
        logits = []

        # h = math.floor(((input_shape[2] - self.filter1[1]) / self.filter1[3] + 1) / 2)
        # w = math.floor(((input_shape[3] - self.filter[2]) / self.filter[4] + 1) / 2)

        while index + self.width_net <= len_input:
            s0 = tf.slice(inputs, [0, 0, 0, index], [-1, -1, -1, self.width_layer])
            s1 = tf.slice(inputs, [0, 0, 0, index + self.width_layer], [-1, -1, -1, self.width_layer])
            s2 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 2], [-1, -1, -1, self.width_layer])
            s3 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 3], [-1, -1, -1, self.width_layer])
            s4 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 4], [-1, -1, -1, self.width_layer])
            # s5 = tf.slice(inputs, [0, 0, 0, index + self.width_layer * 5], [-1, -1, -1, self.width_layer])
            # [?, ?, 80, 20]

            s_fea0 = self.spec_pool(self.spec_conv(s0))
            s_fea1 = self.spec_pool(self.spec_conv(s1))
            s_fea2 = self.spec_pool(self.spec_conv(s2))
            s_fea3 = self.spec_pool(self.spec_conv(s3))
            s_fea4 = self.spec_pool(self.spec_conv(s4))
            # s_fea5 = self.spec_pool(self.spec_conv(s5))
            # [?, ?, 19, 4]

            f1 = self.lay1(inputs=tf.concat([s_fea0, s_fea1], axis=1))
            f2 = self.lay2(inputs=tf.concat([f1, s_fea2], axis=1))
            f3 = self.lay3(inputs=tf.concat([f2, s_fea3], axis=1))
            f4 = self.lay4(inputs=tf.concat([f3, s_fea4], axis=1))
            # f5 = self.lay5(inputs=tf.concat([f4, s_fea5], axis=1))
            # [?, ?, 19, 4]

            if not is_train and index + self.width_net + self.strides > len_input:
                # 此次是最后一个时间步
                self.draw_hid_features(inputs=inputs, h_fea=tf.concat([f1, f2, f3, f4], axis=-1))

            logits.append(f4)
            index += self.strides

        # print(logits[-1].get_shape().as_list())
        # return self.gap(logits[-1])
        return self.dense(self.flatten(logits[-1]))

    def draw_hid_features(self, inputs, h_fea):
        """
        绘制中间层的特征图，保存在本地/hid_pic，第120-121行调用
        :param inputs: [?, 1, 80, 600]
        :param h_fea: [?, 8, 19, 16]
        """
        inputs = np.squeeze(inputs)  # [?, 80, 600]
        h_fea = h_fea.numpy()

        index_sample = 0
        for sample in h_fea:
            # [8, 19, 16]

            yuan_tu = inputs[index_sample]
            # yuan_tu = np.hstack(yuan_tus)

            t_s = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            save_dir = 'hid_pic/' + t_s + '/batch_' + str(self.i) + '/' + str(index_sample)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            Image.fromarray(yuan_tu).convert('RGB').save(save_dir + '/' + 'yuan_tu.jpg')

            index_channel = 0
            for feature in sample:
                # [19, 20]
                save_path = save_dir + '/' + str(index_channel) + '.jpg'
                # scipy.misc.imsave(save_path, feature.T)
                feature = np.array((feature - np.min(feature)) / (np.max(feature) - np.min(feature)) * 255, dtype=int)
                Image.fromarray(feature.T).convert('L').save(save_path)
                index_channel += 1
            index_sample += 1

        self.i += 1
