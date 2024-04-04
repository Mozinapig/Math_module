# -*- encoding: utf-8 -*-
"""
@Modify Time : 2024/4/4 19:01      
@Author : Mozinapig   
@Version : 1.0  
@Desciption : None
  
"""
#0-1整数规划,匈牙利算法
import numpy as np
from scipy.optimize import linear_sum_assignment

cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
row_ind, col_ind = linear_sum_assignment(cost)
print(row_ind)  # 开销矩阵对应的行索引
print(col_ind)  # 对应行索引的最优指派的列索引
print(cost[row_ind, col_ind])  # 提取每个行索引的最优指派列索引所在的元素，形成数组
print(cost[row_ind, col_ind].sum())  # 数组求和

# 输出：
# [0 1 2]
# [1 0 2]
# [1 2 2]
# 5




