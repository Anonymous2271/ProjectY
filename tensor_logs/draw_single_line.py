# coding: utf-8
# ---
# @File: draw_single_line.py
# @description: 用 matplotlib 画折线图，从读取 tensorboard 保存的数据，其在 eager_main 中保存
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 6月10, 2019
# ---

from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensor_logs.line_smooth import smooth
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'Times New Roman'

# # 加载tensorboard日志数据
# ea = event_accumulator.\
#     EventAccumulator('D:/GitHub/ProjectX/tensor_logs/2019-06-14-11-39-58/train/events.out.tfevents.1560483598.localhost.localdomain.v2')
# ea.Reload()
#
# print(ea.scalars.Keys())
#
# line_name = 'loss'
# line = ea.scalars.Items(line_name)
# print(len(line))


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
                data_m[i_line].append(v)
        i_line += 1

    file.close()
    return data_m


index_line = 0  # train loss acc / test loss acc
line_name = 'loss'  # Loss, Accuracy
data = txt_read('lines2020-04-30-01-27-08.txt')

y = [float(i) for i in data[index_line]]
x = range(len(y))

# 平滑处理
# y = smooth(y, 0.99)


fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)

# ax1.plot([float(i) for i in range(length)], [float(i) for i in data[index_line]], label='Train')
ax1.plot(x, y, label='Train')

# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
ax1.set_xlabel("Step")
ax1.set_ylabel(line_name)

plt.legend(loc='lower right')
plt.show()
