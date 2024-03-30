# 顺序号作为秩
import pandas as pd  # 导入用于数据处理的pandas库
import numpy as np  # 导入用于数值运算的numpy库
import statsmodels.api as sm  # 导入用于统计建模的statsmodels库
from scipy.stats import norm  # 从scipy.stats导入正态分布函数

# 定义一个名为'rsr'的函数，用于秩和比率分析
def rsr(data, weight=None, threshold=None, full_rank=True):
    Result = pd.DataFrame()  # 创建一个空的DataFrame以存储结果
    n, m = data.shape  # 获取输入数据的行数（n）和列数（m）

    # 对原始数据进行排名
    if full_rank:
        # 如果full_rank为True，使用密集排名
        for i, X in enumerate(data.columns):
            Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]  # 存储原始数据
            Result[f'R{str(i + 1)}：{X}'] = data.iloc[:, i].rank(method="dense")  # 使用密集排名方法对数据进行排名
    else:
        # 如果full_rank为False，使用替代排名方法
        for i, X in enumerate(data.columns):
            Result[f'X{str(i + 1)}：{X}'] = data.iloc[:, i]  # 存储原始数据
            # 使用替代方法对数据进行排名
            Result[f'R{str(i + 1)}：{X}'] = 1 + (n - 1) * (data.iloc[:, i].max() - data.iloc[:, i]) / (data.iloc[:, i].max() - data.iloc[:, i].min())

    # 计算秩和比率
    weight = 1 / m if weight is None else np.array(weight) / sum(weight)  # 确定秩的权重
    Result['RSR'] = (Result.iloc[:, 1::2] * weight).sum(axis=1) / n  # 计算RSR值
    Result['RSR_Rank'] = Result['RSR'].rank(ascending=False)  # 对RSR值进行排名

    # 创建RSR分布表
    RSR = Result['RSR']
    RSR_RANK_DICT = dict(zip(RSR.values, RSR.rank().values))
    Distribution = pd.DataFrame(index=sorted(RSR.unique()))
    Distribution['f'] = RSR.value_counts().sort_index()  # 每个RSR值的频率
    Distribution['Σ f'] = Distribution['f'].cumsum()  # 累积频率
    Distribution[r'\bar{R} f'] = [RSR_RANK_DICT[i] for i in Distribution.index]  # 每个RSR值的平均秩
    Distribution[r'\bar{R}/n*100%'] = Distribution[r'\bar{R} f'] / n  # 平均秩的百分比
    Distribution.iat[-1, -1] = 1 - 1 / (4 * n)  # 调整最后一个平均秩的百分比
    Distribution['Probit'] = 5 - norm.isf(Distribution.iloc[:, -1])  # 计算Probit值

    # 执行回归分析并计算回归方差
    r0 = np.polyfit(Distribution['Probit'], Distribution.index, deg=1)  # 拟合线性回归
    # 打印回归分析的摘要
    print(sm.OLS(Distribution.index, sm.add_constant(Distribution['Probit'])).fit().summary())
    if r0[1] > 0:
        # 打印带有正截距的回归线方程
        print(f"\n回归直线方程为：y = {r0[0]} Probit + {r0[1]}")
    else:
        # 打印带有负截距的回归线方程
        print(f"\n回归直线方程为：y = {r0[0]} Probit - {abs(r0[1])}")

    # 用回归方程代入并按水平排序
    Result['Probit'] = Result['RSR'].apply(lambda item: Distribution.at[item, 'Probit'])  # 计算RSR的Probit值
    Result['RSR Regression'] = np.polyval(r0, Result['Probit'])  # 计算RSR的回归值
    threshold = np.polyval(r0, [2, 4, 6, 8]) if threshold is None else np.polyval(r0, threshold)  # 设置阈值
    # 基于回归值将结果分类成水平
    Result['Level'] = pd.cut(Result['RSR Regression'], threshold, labels=range(len(threshold) - 1, 0, -1))

    return Result, Distribution

# 定义一个名为'rsrAnalysis'的函数，用于进行RSR分析并将结果输出到Excel文件
def rsrAnalysis(data, file_name=None, **kwargs):
    Result, Distribution = rsr(data, **kwargs)  # 执行RSR分析
    file_name = 'RSR 分析结果报告.xlsx' if file_name is None else file_name + '.xlsx'  # 设置默认文件名
    Excel_Writer = pd.ExcelWriter(file_name)  # 创建Excel写入器
    Result.to_excel(Excel_Writer, '综合评价结果')  # 将综合评价结果写入Excel
    Result.sort_values(by='Level', ascending=False).to_excel(Excel_Writer, '分档排序结果')  # 将排序后的结果写入Excel
    Distribution.to_excel(Excel_Writer, 'RSR分布表')  # 将RSR分布表写入Excel
    Excel_Writer.save()  # 保存Excel文件

    return Result, Distribution