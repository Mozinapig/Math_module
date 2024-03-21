# 通过构造n+1个插值节点来构造n次多项式
# 拉格朗日基函数，拉格朗日插值函数
import sympy as sp


def get_li_numerator(list_x):
    """ 计算li(x)的分子部分，返回分子的表达式 """
    x = sp.symbols('x')
    numerator = [1 for _ in range(0, len(list_x))]  # 初始化分子列表，全置为1
    for i in list(range(0, len(list_x))):
        for j in list(range(0, len(list_x))):
            # 循环计算到自己时，跳过此次循环
            if j == i:
                continue
            value_item = x - list_x[j]
            numerator[i] = numerator[i] * value_item
    return numerator


def get_li_denominator(list_x):
    """ 计算li(x)的分母部分 """
    denominator = [1 for _ in range(0, len(list_x))]  # 初始化分母列表，全置为1
    for i in list(range(0, len(list_x))):
        for j in list(range(0, len(list_x))):
            # 循环计算到自己时，跳过此次循环
            if j == i:
                continue
            value_item = list_x[i] - list_x[j]
            denominator[i] = denominator[i] * value_item
    return denominator


def get_expression_li(list_x):
    """ 计算li(x)的表达式 """
    # 获取分子表达式
    numerator = get_li_numerator(list_x)
    # 获取分母的数值列表
    denominator = get_li_denominator(list_x)
    # 初始化li的表达式列表
    expression_li = []
    for i in list(range(0, len(numerator))):
        expression_li.append(numerator[i] / denominator[i])
    return expression_li


def get_expression_lagrange(list_fx, list_x):
    """ 获取拉格朗日差值多项式的表达式 """
    # 获取li(x)的表达式
    expression_li = get_expression_li(list_x)
    # 初始化拉格朗日插值多项式为0
    expression_lagrange = 0
    for i in list(range(0, len(list_x))):
        expression_lagrange += expression_li[i] * list_fx[i]
    return expression_lagrange


def get_value_expression_lagrange(expression_lagrange,  # 拉格朗日差值多项式的表达式
                                  x,  # 某一点x
                                  precision=5  # 计算保留的有效数字
                                  ):
    """ 计算拉格朗日差值多项式在某一点x的值 """
    return expression_lagrange.evalf(subs={'x': x}, n=precision)


x = sp.symbols('x')
# 设置原始数据，节点值、节点函数值
list_x = [0, 0.2, 0.4, 0.6, 0.8, 1]
list_fx = [1, 2, 3, 4, 5, 6]
# 设置待预测的点
interpolation_x = 2
# 计算拉格朗日插值多项式
expression_lagrange = get_expression_lagrange(list_fx, list_x)
# 计算该多项式在计算点的值
result_interpolation = get_value_expression_lagrange(expression_lagrange, interpolation_x)
print("拉格朗日插值多项式：", expression_lagrange)
print("插值结果：", result_interpolation)
