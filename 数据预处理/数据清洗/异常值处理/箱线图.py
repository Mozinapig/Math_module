# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def OutlierDetection(df):
    # 计算下四分位数和上四分位
    Q1 = df.quantile(q=0.25)
    Q3 = df.quantile(q=0.75)

    # 基于1.5倍的四分位差计算上下须对应的值
    low_whisker = Q1 - 1.5 * (Q3 - Q1)
    up_whisker = Q3 + 1.5 * (Q3 - Q1)

    # 寻找异常点
    kk = df[(df > up_whisker) | (df < low_whisker)]
    data1 = pd.DataFrame({'id': kk.index, '异常值': kk})
    return data1



if __name__ == '__main__':
    # 创建数据
    data = [1222, 87, 77, 92, 68, 80, 78, 84, 77, 81, 80, 80, 77, 92, 86, 76, 80, 81, 75, 77, 72, 81, 72, 84, 86, 80,
            68, 77, 87, 76, 77, 78, 92, 75, 80, 78, 123, 3, 1223, 1232]
    df = pd.DataFrame(data, columns=['value'])
    df=df.iloc[:,0]
    result=OutlierDetection(df)
    print('箱线图检测到的异常值如下---------------------')
    print(result)
