# coding: utf-8
# ---
# @File: test.py
# @description: RNN模拟
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 3月22, 2020
# ---

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

#
# def gen_data(size=1000000):
#     """
#     生成数据，其规则为
#     输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0;
#     输出数据Y：在实践t，Yt的值有50%的概率为1，50%的概率为0;
#     除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%， 如果`Xt-8 == 1`，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。
#     """
#     X = np.array(np.random.choice(2, size=(size,)))
#     Y = []
#     for i in range(size):
#         threshold = 0.5
#         if X[i-3] == 1:
#             threshold += 0.5
#         if X[i-8] == 1:
#             threshold -= 0.25
#         if np.random.rand() > threshold:
#             Y.append(0)
#         else:
#             Y.append(1)
#     return X, np.array(Y)
#
#
# def gen_batch(raw_data, batch_size, num_steps):
#     # raw_data是使用gen_data()函数生成的数据，分别是X和Y
#     raw_x, raw_y = raw_data
#     data_length = len(raw_x)
#
#     # 首先将数据切分成batch_size份，0-batch_size，batch_size-2*batch_size
#     batch_partition_length = data_length // batch_size
#     data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
#     data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
#     for i in range(batch_size):
#         data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
#         data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
#
#     # 因为RNN模型一次只处理num_steps个数据，所以将每个batch_size在进行切分成epoch_size份，每份num_steps个数据。注意这里的epoch_size和模型训练过程中的epoch不同。
#     epoch_size = batch_partition_length // num_steps
#
#     # x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
#     for i in range(epoch_size):
#         x = data_x[:, i * num_steps:(i + 1) * num_steps]
#         y = data_y[:, i * num_steps:(i + 1) * num_steps]
#         yield (x, y)
#
#
# # 这里的n就是训练过程中用的epoch，即在样本规模上循环的次数
# def gen_epochs(n, num_steps):
#     for i in range(n):
#         yield gen_batch(gen_data(), batch_size, num_steps)


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, unit=32):
        super(MyLayer, self).__init__()
        self.unit = unit

    def build(self, input_shape):
        # self.weight = self.add_weight(shape=(input_shape[-1], self.unit),
        #                               initializer=tf.keras.initializers.RandomNormal(),
        #                               trainable=True)
        # self.bias = self.add_weight(shape=(self.unit,),
        #                             initializer=tf.keras.initializers.Zeros(),
        #                             trainable=True)
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=[3, 3],
                                            strides=[1, 1], padding='valid',
                                            activation=tf.nn.leaky_relu, data_format='channels_first', trainable=True)
        self.dense = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, inputs):
        # a = tf.matmul(inputs, self.weight) + self.bias
        a = self.conv1(inputs)

        return self.dense(a)


class MyBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.layer1 = MyLayer(32)
        self.layer2 = MyLayer(16)
        self.layer3 = MyLayer(2)

    def call(self, inputs):
        h1 = self.layer1(inputs)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2 = tf.nn.relu(h2)
        return self.layer3(h2)


my_block = MyBlock()
print('trainable weights:', len(my_block.trainable_weights))
y = my_block(tf.ones(shape=(10, 3, 64, 64)))
# 构建网络在build()里面，所以执行了才有网络
print('trainable weights:', len(my_block.trainable_weights))
print(my_block.trainable_weights)
