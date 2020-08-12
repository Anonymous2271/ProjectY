# coding: utf-8
# ---
# @File: draw_single_line.py
# @description: 用 matplotlib 画折线图，从读取 tensorboard 保存的数据，其在 eager_main 中保存
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 6月10, 2019
# ---

import matplotlib.pyplot as plt
import matplotlib
from tensor_logs.line_smooth import smooth
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'Times New Roman'


def txt_read(logs_path):
    loss_history = []
    acc_history = []
    test_loss_history = []
    test_acc_history = []
    data_m = [loss_history, acc_history, test_loss_history, test_acc_history]

    file = open(logs_path, 'r')
    i_line = 0
    for line in file.readlines():
        line = line.strip('\n')
        for v in line.split('\t'):
            if v != '':
                data_m[i_line].append(float(v))
        i_line += 1

    file.close()
    return data_m


data = txt_read('lines_0.95_和论文里一样的参数，除了前置卷积.txt')

loss_history = smooth(data[0], weight=0.9)
acc_history = smooth(data[1], weight=0.9)
test_loss_history = smooth(data[2], weight=0.5)
test_acc_history = smooth(data[3], weight=0.5)
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
