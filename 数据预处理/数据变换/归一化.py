##通常我们要将数据全部换为极大型指标

from sklearn import preprocessing
import pandas as pd
import numpy as np

#minmax,mean,z_score三种方法

def MaxMinNormalizetion(x):
    shapeX = x.shape
    rows = shapeX[0]
    cols = shapeX[1]
    headers = list(x)
    result = pd.DataFrame(columns=headers)
    for i in range(0, rows, 1):
        dict1 = {}
        dict1[headers[0]] = x['No'][i]
        for j in range(1, cols, 1):
            maxcol = x[headers[j]].max()
            mincol = x[headers[j]].min()
            val = (x.iloc[i, j] - mincol) / (maxcol - mincol)  # 一般是(x-min)/(max-min)进行归一化处理
            dict1[headers[j]] = val
        result = result.append(dict1, ignore_index=True)
    return result


data1 = pd.read_csv(r'CatInfo.csv')
print('original data\n', data1)
newdata = MaxMinNormalizetion(data1)
print('归一化的数据\n', newdata)

from sklearn import preprocessing
import pandas as pd
import numpy as np

print("去除空值并且归一化处理")
y = data1.dropna(axis=0).iloc[:, 1:]  # 去除空值
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(y)
print(x_minmax)