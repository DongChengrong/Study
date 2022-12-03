import pandas as pd
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def BP(x_train, y_train, numb, inta):  # x_train,y_train表示训练集，numb表示隐层神经元的个数，inta表示学习率
    # print('x_train:',x_train)
    # print('y_train:',y_train)
    v = np.matrix(np.random.rand(len(x_train.T), numb))  # 随机生成输入层神经元与隐层神经元之间的连接权
    # print('v:',v)
    w = np.matrix(np.random.rand(numb, len(y_train.T)))  # 随机生成隐层神经元与输出层神经元之间的连接权
    # print('w:',w)
    thita = np.matrix(np.random.rand(len(y_train.T)))  # 输出神经元的阈值
    # print("thita:",thita)
    garma = np.matrix(np.random.rand(numb))  # 输出神经元的阈值
    # print("garma:",garma)
    temp = 0
    for i in range(len(x_train)):
        alpha = x_train[i].dot(v)  # 隐层神经元的输入
        # print('alpha',alpha)
        b = sigmoid(alpha)  # 隐层神经元的输出
        # print('b',b)
        beta = b.dot(w)  # 输出神经元的输入
        # print('beta',beta)
        y_estimate = sigmoid(beta - thita)  # 求出y的估计值
        # print('y_estimate',y_estimate)
        g = y_estimate.dot((1 - y_estimate).T).dot(y_train[i] - y_estimate)  # 输出层神经元的梯度
        # print('g',g)
        e = b.dot((1 - b).T).dot(g).dot(w.T)  # 隐层神经元的梯度
        # print('e',e)
        E = 1 / 2 * (y_estimate - y_train[i]).dot((y_estimate - y_train[i]).T)  # 均方误差
        # print('E',E)
        if E > temp:  # 更新连接权和阈值
            w = w + inta * (b.T).dot(g)
            # print('w',w)
            thita = thita - inta * g
            # print('thita',thita)
            v = v + inta * (x_train[i].T).dot(e)
            # print('v',v)
            garma = garma - inta * e  # 隐层神经元的阈值
            # print('garma',garma)
            temp = E

    return w, v, thita, garma


x = np.matrix(np.random.rand(3, 4))  # 输入层：随机生成3个样本，每个样本有4个神经元
y = np.matrix(np.random.rand(3, 2))  # 输出层：随机生成3个样本，每个样本有2个神经元
print('x', x)
print('y', y)

w, v, thita, garma = BP(x, y, 4, 0.01)  # 这里我假设隐层有4个神经元，学习率为0.01
print('w', w)
print('v', v)
print('thita', thita)
print('garma', garma)
