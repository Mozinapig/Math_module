# 基于3sigma的异常值检测
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 导入绘图库


##3sigma原则
def threesigma(data, n):
    '''
    data：表示时间序列，包括时间和数值两列；
    n：表示几倍的标准差
    '''
    data_x = data.ix[:, 0].tolist()  ##获取时间序列的时间
    # print (data_x)
    # print ("**********",j)
    mintime = data_x[0]  ##获取时间序列的起始年份
    maxtime = data_x[-1]  ##获取时间序列的结束年份

    data_y = data.ix[:, 1].tolist()  ##获取时间序列数值
    ymean = np.mean(data_y)  ##求时间序列平均值
    ystd = np.std(data_y)  ##求时间序列标准差
    down = ymean - n * ystd  ##计算下界
    up = ymean + n * ystd  ##计算上界

    outlier = []  # 将异常值保存
    outlier_x = []

    for i in range(0, len(data_y)):
        if (data_y[i] < down) | (data_y[i] > up):
            outlier.append(data_y[i])
            outlier_x.append(data_x[i])
        else:
            continue
    return mintime, maxtime, outlier, outlier_x


# 设置列表，用于记录结果
indicator = []  ##指标
flag = []  ##是否为异常值
outlier_data = []  ##异常值
outlier_time = []  ##出现异常值的对应时间
max_time = []  ##时间序列的开始时间
min_time = []  ##时间序列的结束时间
time_flag = []  ##异常值是否为起始时间

#读取数据
data = pd.read_csv('data.csv', index_col=False, encoding='gb18030')
# print(data.head())
col_name = data.columns.tolist()
# print (data.columns.tolist())

# 设置参数
n = 3  # n*sigma
print("******************:n=", n)

##依次检测每一个指标
for j in col_name[1:]:
    indicator.append(j)
    temp_data = data.ix[:, ['时间', j]]
    # print ("删除空值前",len(temp_data))
    temp_data = temp_data.dropna()  # 删除空值
    # print("删除空值后",len(temp_data))
    temp_data = temp_data.sort_values(by=['时间'], axis=0, ascending=True)  # 按时间排序
    # print (temp_data)

    mintime, maxtime, outlier, outlier_x = threesigma(temp_data, n)  # 调用3sigma函数
    min_time.append(mintime)  ##获取时间序列的起始年份
    max_time.append(maxtime)  ##获取时间序列的结束年份

    outlier_data.append(outlier)
    outlier_time.append(outlier_x)

    if (maxtime in outlier_x) or (mintime in outlier_x):
        time_flag.append('异常值为起始端')
        # print (time_flag)
    else:
        time_flag.append("")

    if len(outlier) > 0:
        flag.append('异常')
        print("***************************")
        print("异常指标：", j)
        print('\n异常数据如下：', outlier, outlier_x)
        ##画出存在异常值的时间序列的折线图，异常值处特殊标注
        plt.figure(figsize=(12, 5))
        plt.plot(temp_data.ix[:, 0], temp_data.ix[:, 1])
        plt.plot(outlier_x, outlier, 'ro')
        for j in range(len(outlier)):
            plt.annotate(outlier[j], xy=(outlier_x[j], outlier[j]), xytext=(outlier_x[j], outlier[j]))
        plt.show()

    else:
        flag.append('正常')

result = pd.DataFrame()
result['指标'] = indicator
result['开始时间'] = min_time
result['结束时间'] = max_time
result['是否异常'] = flag
result['异常数值'] = outlier_data
result['异常时间'] = outlier_time
result['异常值所处时间标识'] = time_flag
result
