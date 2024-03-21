# 此处只记录二维插值的三维图
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def func(x, y):
    return (x + y) * np.exp(-5.0 * (x ** 2 + y ** 2))


# X-Y轴分为20*20的网格
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
x, y = np.meshgrid(x, y)  # 20*20的网格数据

fvals = func(x, y)  # 计算每个网格点上的函数值  15*15的值

fig = plt.figure(figsize=(18,10))  # 设置图的大小
# Draw sub-graph1
ax = plt.subplot(1, 2, 1, projection='3d')  # 设置图的位置
surf = ax.plot_surface(x, y, fvals, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5,
                       antialiased=True)  # 第四个第五个参数表示隔多少个取样点画一个小面，第六个表示画图类型，第七个是画图的线宽，第八个表示抗锯齿
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')  # 标签
plt.colorbar(surf, shrink=0.5, aspect=5)  # 标注

# 二维插值
newfunc = interpolate.interp2d(x, y, fvals, kind='cubic')  # newfunc为一个函数

# 计算100*100的网格上的插值
xnew = np.linspace(-1, 1, 100)  # x
ynew = np.linspace(-1, 1, 100)  # y
fnew = newfunc(xnew, ynew)  # 仅仅是y值   100*100的值  np.shape(fnew) is 100*100
print(fnew)
xnew, ynew = np.meshgrid(xnew, ynew)
ax2 = plt.subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xnew, ynew, fnew, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax2.set_xlabel('xnew')
ax2.set_ylabel('ynew')
ax2.set_zlabel('fnew(x, y)')
plt.colorbar(surf2, shrink=0.5, aspect=5)  # 标注
plt.show()

