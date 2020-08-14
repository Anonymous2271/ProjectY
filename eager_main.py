# coding: utf-8
# ---
# @File: eager_main.py
# @description: 主函数，使用 tensorflow eager 模式
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 3月18, 2019
# ---

import tensorflow as tf
import numpy as np
import time
from input_data import BatchGenerator
from MyException import MyException
# from y_model import YModel
from y_model_v2 import YModel
from base_line import CNN_GRU
from tensor_logs.line_smooth import smooth
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.size'] = 18
tf.keras.backend.clear_session()

# model para
learning_rate = 0.005
decay = 0.0001  # 学习率衰减
momentum = 0.7  # 动量
clip_norm = 1.5  # 梯度裁剪阈值

n_layers = 5  # 网络的层数
width_layer = 6  # 网络每层的宽度
strides = 3  # 网络在序列数据中每个时间步前进的步长
batch_size = 32
epoch = 12  # 训练的 epoch 数，从1开始计数
rate_subset = 1  # 使用的子集占整个数据集的比例，为1时使用全部数据集
rate_test = 0.3  # 测试数据占使用数据的比例
n_classes = 8
is_train = True

# data to store
loss_history = []
acc_history = []
test_loss_history = []
test_acc_history = []
y_true = []
y_pred = []


def txt_save(data_m, name):
    """
    以 txt 保存实验日志，Tensorboard 输出的图不够专业，只好自己写一个；
    该 txt 文件，在draw_cm.py, draw_many_line.py, draw_single_line.py 中均支持，txt 文件说明如下：
    ————————————————————
    训练损失------line
    训练准确率----line
    测试损失------line
    测试准确率----line
    ————————————————————
    :param data_m: 数据list
    :param name: 文件名
    """
    logs_path = 'tensor_logs/' + time.strftime(name + "%Y-%m-%d-%H-%M-%S", time.localtime()) + '.txt'
    # logs_path = 'tensor_logs/' + name + "_over5" + '.txt'
    file = open(logs_path, 'a')
    for line in data_m:
        for v in line:
            s = str(v) + '\t'
            file.write(s)
        file.write('\n')
    file.close()
    print(name + 'saved')


# 初始化 input_data 类的对象
batch_generator = BatchGenerator(file_dir='D:/dataset/new_images', n_classes=n_classes, rate_subset=rate_subset,
                                 rate_test=rate_test, is_one_hot=False, data_format='channels_first')


def cal_loss(logits, lab_batch):
    """
    计算损失
    :param logits: 模型输出
    :param lab_batch: 标签 batch
    :return: loss，tensor 类型
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab_batch, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss


# 初始化模型和优化器
the_model = YModel(n_classes=n_classes, n_layers=n_layers, width_layer=width_layer, strides=strides)
# the_model = CNN_GRU(n_classes=n_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# 在这里得到的参数是0，因为自定义的网络，在build()执行之后才会有graph
# trainable_vas = the_model.trainable_weights
# print(trainable_vas, 'trainable_vas')

step = 1  # 训练step，一个 step 处理一个 batch 的数据
try:
    while True:  # 从训练到测试的节奏由 batch_generator 控制，因此写个无限循环就行
        batch_x, batch_y, epoch_index = batch_generator.next_batch(batch_size=batch_size, epoch=epoch)
        # learning_rate = my_learning_rate(epoch_index, step)
        if epoch_index != 0:
            is_train = True
            # 判定训练
        else:
            is_train = False
            # 判定测试

        # 记录梯度
        with tf.GradientTape() as tape:
            logits = the_model(inputs=batch_x, is_train=is_train)
            loss = cal_loss(logits, batch_y)

        # 如果为训练阶段，则应用梯度下降，让模型学习；测试阶段什么都不做
        if epoch_index != 0:
            # print(len(the_model.trainable_weights))

            grads = tape.gradient(loss, the_model.trainable_weights)
            # grads, _ = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
            optimizer.apply_gradients(zip(grads, the_model.trainable_weights))
        else:
            pass

        # 计算准确率
        correct_pred = tf.equal(tf.argmax(logits, 1), batch_y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        step += 1  # step 自增
        print('epoch:{}, step:{}, loss:{:.4f}, acc:{:.4f}'.
              format(epoch_index, step, loss, accuracy))

        # 记录每个 step 的实验日志
        if epoch_index != 0:
            loss_history.append(loss.numpy())
            acc_history.append(accuracy.numpy())
        else:
            test_loss_history.append(loss.numpy())
            test_acc_history.append(accuracy.numpy())
            #
            #     # 测试阶段，选择最好的一个批次，记录预测值和标签值，用于混淆矩阵分析
            #     # if best_acc < accuracy.numpy():
            #     #     y_pred = tf.math.argmax(logits, axis=1).numpy()
            #     #     y_true = tf.math.argmax(batch_y, axis=1).numpy()
            #     #     best_acc = accuracy.numpy()
            # 测试阶段，记录全部批次的记录预测值和标签值，用于混淆矩阵分析
            for l in tf.math.argmax(logits, axis=1).numpy():
                y_pred.append(l)
            for y in batch_y:
                y_true.append(y)

# 捕获 input_data 在数据输送结束时的异常，开始画图
except MyException as e:
    # 先保存日志文件
    data_m = [loss_history, acc_history, test_loss_history, test_acc_history]
    txt_save(data_m, name='lines')
    txt_save([y_pred, y_true], name='y_')

    # 平均准确率, 损失, 方差
    print('acc [mean, variance]: ', np.mean(test_acc_history), np.var(test_acc_history))
    print('loss [mean, variance]: ', np.mean(test_loss_history), np.var(test_loss_history))

    # 折线平滑
    acc_history = smooth(acc_history, weight=0.9)
    loss_history = smooth(loss_history, weight=0.9)
    # test_acc_history = smooth(test_acc_history, weight=0.8)
    # test_loss_history = smooth(test_loss_history, weight=0.8)

    # 画图
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    colors1 = 'C0'
    colors2 = 'C1'

    axs[0, 0].plot(acc_history, label='train', color=colors1)
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].set_xlabel('step')
    axs[0, 0].set_ylabel('accuracy')

    axs[0, 1].plot(loss_history, label='train', color=colors1)
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].set_xlabel('step')
    axs[0, 1].set_ylabel('loss')

    axs[1, 0].plot(test_acc_history, label='test', color=colors1)
    axs[1, 0].legend(loc='lower right')
    axs[1, 0].set_xlabel('step')
    axs[1, 0].set_ylabel('accuracy')

    axs[1, 1].plot(test_loss_history, label='test', color=colors1)
    axs[1, 1].legend(loc='lower right')
    axs[1, 1].set_xlabel('step')
    axs[1, 1].set_ylabel('loss')

    plt.show()
    pass
