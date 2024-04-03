# -*- encoding: utf-8 -*-
"""
@Modify Time : 2024/4/3 19:43      
@Author : Mozinapig   
@Version : 1.0  
@Desciption : None
  
"""
# -*- coding: utf-8 -*-
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers.core import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical

warnings.filterwarnings("ignore")

np.random.seed(2222)


# 3 建立一个简单BP神经网络模型

def training(X):
    neurons1 = int(X[0])
    neurons2 = int(X[1])
    dropout = round(X[2], 6)
    batch_size = int(X[3])
    print(X)
    # nor开头的样本取全部  剩下的取0.1  0.05   0.02模拟样本不平衡  一共要生成三个数据集
    nor_file = 'data/normal.xls'
    other_dir = 'data\\'
    files = os.listdir(other_dir)
    i = 0.02
    data = pd.read_excel(nor_file, sheet_name=0)  # pandas以DataFrame的格式读入excel表
    data.loc[:, 'label'] = 0  # 设置nor开头的标签为0
    for file in files:
        if file != 'normal.xls':
            data_other_all = pd.read_excel(other_dir + file, sheet_name=0)  # pandas以DataFrame的格式读入excel表
            data_other01 = data_other_all.sample(int(i * data.shape[0]))
            data_other01.loc[:, 'label'] = files.index(file) + 1  # 设置非正常的标签为1-7
            data = pd.concat([data, data_other01], axis=0)

    feature = ['TEI', 'TEO', 'TCI', 'TCO', 'TRE', 'TRC', 'kW', 'TRC_sub', 'Tsh_suc', 'PO_feed',
               'TCA']  # 影响因素11个
    label = ['label']  # 标签一个，即需要进行预测的值
    # 2 数据预处理和标注
    data_mean = data.mean()
    data_std = data.std()
    data_train = (data - data_mean) / data_std  # 数据标准化
    x_train = data[feature].values  # 特征数据
    y_train = to_categorical(data[label])  # 标签数据

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.2)
    model = Sequential()  # 层次模型
    model.add(Dense(neurons1, activation='relu', input_dim=11))  # 输入层，Dense表示BP层
    model.add(Dropout(dropout))
    model.add(Dense(neurons2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    # 用focal_loss做损失函数,希望准确率和召回率比使用交叉熵损失函数高
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')  # 编译模型
    history = model.fit(x_train, y_train, epochs=1500, batch_size=x_train.shape[0], verbose=0)

    # 4 预测，并还原结果。
    y_pre = model.predict(x_valid)
    # 准确率和召回率
    y_valid = y_valid.round()
    y_pre = y_pre.round()
    print('剩下的取  ', str(i))
    print("整体准确率：", accuracy_score(y_valid, y_pre))
    report = classification_report(y_valid.round(), y_pre.round(), output_dict=True)
    # 计算其他类别的召回率
    # print('其他召回率：', report['8']['recall'])
    print(report)
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=22,
        validation_split=0.1,
        verbose=1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=22, restore_best_weights=True)])

    pred = model.predict(x_valid)
    le = len(pred)
    y_t = y_valid.reshape(-1, 1)
    return pred, le, y_t


def function(ps, test, le):
    ss = sum(((abs(test - ps)) / test) / le)
    return ss


# (1) PSO Parameters
MAX_EPISODES = 22
MAX_EP_STEPS = 22
c1 = 2
c2 = 2
w = 0.5
pN = 100  # 粒子数量

# (2) LSTM Parameters
dim = 4  # 搜索维度
X = np.zeros((pN, dim) )  # 所有粒子的位置和速度
V = np.zeros((pN, dim))
pbest = np.zeros((pN, dim))  # 个体经历的最佳位置和全局最佳位置
gbest = np.zeros(dim)
p_fit = np.zeros(pN)  # 每个个体的历史最佳适应值
print(p_fit.shape)
print(p_fit.shape)
t1 = time.time()

'''
神经网络第一层神经元个数
神经网络第二层神经元个数
dropout比率
batch_size
'''
UP = [150, 15, 0.5, 16]
DOWN = [50, 5, 0.05, 8]

# (4) 开始搜索
for i_episode in range(MAX_EPISODES):
    """初始化s"""
    random.seed(8)
    fit = -1e5  # 全局最佳适应值
    # 初始粒子适应度计算
    print("计算初始全局最优")
    for i in range(pN):
        for j in range(dim):
            V[i][j] = random.uniform(0, 1)
            if j == 2:
                X[i][j] = random.uniform(DOWN[j], UP[j])
            else:
                X[i][j] = round(random.randint(DOWN[j], UP[j]), 0)
        pbest[i] = X[i]
        le, pred, y_t = training(X[i])
        NN = 1
        # 计算适应值
        tmp = function(pred, y_t, le)
        p_fit[i] = tmp
        if tmp > fit:
            fit = tmp
            gbest = X[i]
    print("初始全局最优参数：{:}".format(gbest))

    fitness = []  # 适应度函数
    for j in range(MAX_EP_STEPS):
        fit2 = []
        plt.title("第{}次迭代".format(i_episode))
        for i in range(pN):
            le, pred, y_t = training(X[i])
            temp = function(pred, y_t, le)
            fit2.append(temp / 1000)
            if temp > p_fit[i]:  # 更新个体最优
                p_fit[i] = temp
                pbest[i] = X[i]
                if p_fit[i] > fit:  # 更新全局最优
                    gbest = X[i]
                    fit = p_fit[i]
        print("搜索步数：{:}".format(j))
        print("个体最优参数：{:}".format(pbest))
        print("全局最优参数：{:}".format(gbest))
        for i in range(pN):
            V[i] = w * V[i] + c1 * random.uniform(0, 1) * (pbest[i] - X[i]) + c2 * random.uniform(0, 1) * (
                    gbest - X[i])
            ww = 1
            for k in range(dim):
                if DOWN[k] < X[i][k] + V[i][k] < UP[k]:
                    continue
                else:
                    ww = 0
            X[i] = X[i] + V[i] * ww
        fitness.append(fit)

print('Running time: ', time.time() - t1)
