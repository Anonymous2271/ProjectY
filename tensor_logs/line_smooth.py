# coding: utf-8
# ---
# @File: line_smooth.py
# @description:
# @Author: Xin Zhang
# @E-mail: meetdevin.zh@outlook.com
# @Time: 4月29, 2020
# ---


def smooth(values, weight=0.5):
    """
    :param values: 点集合
    :param weight: 平滑度
    :return:
    """
    last = values[0]  # 上一个值
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point  # 这个方程是从 Tensorboard 源码中扒出来的
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def line_fit(x, y):
    # 用最小二乘法求点的拟合曲线，没上面那个函数好用，会改变曲线轨迹
    y = list(map(float, y))
    parameter = np.polyfit(x=x, y=y, deg=3)  # 输出三次方程的参数
    p = np.poly1d(parameter)  # 根据参数输出方程
    print(p)
    return p(x)