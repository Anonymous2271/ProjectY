# coding: utf-8
# ---
# @File: y_layer.py
# @description: 
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 4月04, 2020
# ---

import tensorflow as tf
import tensorflow.keras as keras


class LinkMemoryLayer(keras.layers.Layer):
    def __init__(self, filter_para=None, my_name=None):
        """
        :param filter_para:
        :param my_name:
        """
        super(LinkMemoryLayer, self).__init__()
        if filter_para is None:
            filter_para = [8, 3, 3, 1, 1]
        self.f_para = filter_para
        self.my_name = my_name

    def build(self, input_shape):
        # [?, filter[0]*2, h, w]
        h = input_shape[2]
        # w = int(input_shape[3]/2)
        w = input_shape[3]
        self.batch_size = input_shape[0]  # 这里的input_shape只取了第一次的，当最后一个批次的size有变化时，可能出错。

        self.conv_relation = keras.layers.Conv2D(filters=self.f_para[0], kernel_size=[self.f_para[1], self.f_para[2]],
                                                 strides=[self.f_para[3], self.f_para[4]], padding='same',
                                                 activation=None, data_format='channels_first',
                                                 kernel_initializer=keras.initializers.glorot_uniform,
                                                 bias_initializer=keras.initializers.zeros)
        # self.pool_relation = keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='valid',
        #                                             data_format='channels_first')
        self.batch_norm = keras.layers.BatchNormalization()
        self.activate = keras.layers.LeakyReLU()
        self.Memory = self.add_weight(shape=[self.batch_size, self.f_para[0], h, w],
                                      initializer=keras.initializers.orthogonal, trainable=False)
        # 版本 1 的参数 #################
        # self.U_weight = self.add_weight(shape=[1, self.f_para[0], h, h],
        #                                 initializer=keras.initializers.glorot_uniform())
        # self.V_weight = self.add_weight(shape=[1, self.f_para[0], h, h],
        #                                 initializer=keras.initializers.glorot_uniform())
        # 版本 2 and 3 的参数 #################
        # self.conv_gate = keras.layers.Conv2D(filters=self.f_para[0], kernel_size=[w, w], strides=[1, 1],
        #                                      padding='valid', activation=tf.nn.leaky_relu, data_format='channels_first')
        # self.pool_gate = keras.layers.MaxPool1D(pool_size=2, strides=2, padding='valid',
        #                                         data_format='channels_first')
        # self.flatten = keras.layers.Flatten(data_format='channels_first')
        # self.dense_gate = keras.layers.Dense(units=self.f_para[0] * h * w, activation=tf.nn.sigmoid)

        # self.deconv_gate = keras.layers.Conv2DTranspose(filters=self.f_para[0], kernel_size=[w, w], strides=[1, 1],
        #                                                 padding='valid', activation=tf.nn.sigmoid, data_format='channels_first')
        # 版本 4  的参数 ####################
        self.conv_gate = keras.layers.Conv2D(filters=self.f_para[0], kernel_size=[5, 5], strides=[1, 1], padding='same',
                                             activation=tf.nn.sigmoid, data_format='channels_first',
                                             kernel_initializer=keras.initializers.glorot_uniform,
                                             bias_initializer=keras.initializers.zeros)

    def call(self, inputs, **kwargs):
        # [?, filter[0], h, w]
        feat_relation = self.conv_relation(inputs)
        feat_relation = self.batch_norm(feat_relation)
        feat_relation = self.activate(feat_relation)
        # [?, ?, 38, 8]
        ################################
        # 版本 1 ：使用 GRU 的门控逻辑，优点是1)参数少，需要训练的参数就两个；2)可以做到与卷积通道相对应
        ################################
        # u_tiled = tf.tile(self.U_weight, [self.batch_size, 1, 1, 1], name="u_tiled")  # [bs, c, h, w]
        # v_tiled = tf.tile(self.V_weight, [self.batch_size, 1, 1, 1], name="v_tiled")  # [bs, c, h, w]
        # gate = tf.nn.sigmoid(tf.matmul(u_tiled, feat_relation) + tf.matmul(v_tiled, self.Memory))

        ################################
        # 版本 2 and 3 ：使用全连接层/反卷积作为门控逻辑
        ################################
        # feat_pool = self.pool_relation(feat_relation)
        # [?, ?, 19, 4]
        #
        # fea_vector = self.conv_gate(tf.concat([feat_relation, self.Memory], axis=1))
        # 版本 2 ####
        # fea_vector = self.pool_gate(tf.squeeze(fea_vector))
        # fea_vector = self.flatten(fea_vector)
        # gate_vector = self.dense_gate(fea_vector)
        # gate = tf.reshape(gate_vector, shape=feat_relation.get_shape().as_list())
        # 3 ####
        # gate = self.deconv_gate(fea_vector)
        ################################
        # 版本 4 ：使用卷积
        ################################
        gate = self.conv_gate(tf.concat([feat_relation, self.Memory], axis=1))

        # 门控
        self.Memory = tf.multiply(gate, self.Memory) + tf.multiply(1 - gate, feat_relation)
        self.Memory = tf.nn.tanh(self.Memory)
        return self.Memory
