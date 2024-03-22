#1.数据预处理，极大化，无量纲化，向量规范化
#2.规范化处理完毕后，进行加权规范矩阵运算
#3.确定正理想解和负理想解
#4.求距离并计算排队指示值

#这里记录  随机熵权TOPSIS

import numpy as np
import pandas as pd
# 熵权法计算权重
def entropy_weight(data):
    # 数据标准化
    data = np.array(data)
    data = data / data.sum(axis=0)

    # 计算熵值
    k = 1.0 / np.log(len(data))
    entropy = -k * np.nansum(data * np.log(data + 1e-12), axis=0)  # 避免对零取对数
    weight = (1 - entropy) / (1 - entropy).sum()
    return weight

# 定义一个函数来处理区间型变量
def normalize_interval_type(x, a, b, a_star, b_star):
    """
    对于区间型变量，根据提供的图形公式进行处理
    """
    if x < a:
        return 1 - (a - x) / (a - a_star)
    elif a <= x <= b:
        return 1
    elif x > b:
        return 1 - (x - b) / (b_star - b)

# 修改后的TOPSIS算法
def topsis(data, weights, criteria, a_star, b_star):
    # 数据归一化
    norm_data = np.sqrt(data**2 / (data**2).sum(axis=0))

    # 处理不同类型的指标
    for i, crit in enumerate(criteria):
        if crit == 'cost':  # 越小越好
            norm_data[:, i] = 1 - norm_data[:, i]
        elif crit == 'nominal':  # 区间型
            a = norm_data[:, i].min()  # 最小值
            b = norm_data[:, i].max()  # 最大值
            norm_data[:, i] = np.array([normalize_interval_type(x, a, b, a_star, b_star) for x in norm_data[:, i]])

    # 理想最优解和最劣解
    positive_ideal = norm_data.max(axis=0)
    negative_ideal = norm_data.min(axis=0)

    # 计算距离
    pos_distance = np.sqrt(((norm_data - positive_ideal)**2 * weights).sum(axis=1))
    neg_distance = np.sqrt(((norm_data - negative_ideal)**2 * weights).sum(axis=1))

    # 得分计算
    score = neg_distance / (pos_distance + neg_distance)
    return score

# 生成随机权重集
def generate_random_weights(base_weights, num_sets=100, max_range=0.5):
    max_random = max_range * max(base_weights)
    random_weight_sets = []
    for _ in range(num_sets):
        random_weights = base_weights + np.random.rand(len(base_weights)) * max_random
        random_weights /= random_weights.sum()  # 归一化
        random_weight_sets.append(random_weights)
    return random_weight_sets

# 示例数据
file_path = 'a.xlsx'  # 请替换为您的文件路径
data_df = pd.read_excel(file_path)

# 将数据转换为numpy数组
data = data_df.to_numpy()

# 计算基准权重
base_weights = entropy_weight(data)

# 生成随机权重集
random_weight_sets = generate_random_weights(base_weights, num_sets=100)

# 示例criteria，根据实际情况定制
criteria = ['benefit', 'cost','benefit','benefit','benefit','cost']
a_star = 0.3  # 区间型指标最佳下限，根据实际情况定制
b_star = 0.7  # 区间型指标最佳上限，根据实际情况定制

# 应用TOPSIS算法并计算平均得分和方差
scores = np.array([topsis(data, weights, criteria, a_star, b_star) for weights in random_weight_sets])
average_scores = scores.mean(axis=0)
variance = scores.var(axis=0)
print(f"平均得分为{average_scores}")
# 排序
sorted_indices = np.argsort(-average_scores)  # 得分按降序排序
sorted_scores = average_scores[sorted_indices]
sorted_indices