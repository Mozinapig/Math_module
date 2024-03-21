# 导入库和数据
import numpy as np
from scipy import interpolate
import pylab as plb

x0 = np.linspace(0, 10, 11)
y = (np.cos(x0) + np.sin(x0)) * np.exp(x0 * 0.1)
plb.plot(x0, y, 'ro')
x1 = np.linspace(0, 10, 101)

# 插值方法：1.阶梯插值 2.线性插值 3.2阶样条插值 4.3阶样条插值
# "nearest"和"zero"==阶梯插值
# "slinear"=线性插值
# "quadratic"==2阶样条插值
# "cubic"==3阶样条插值
style = ["nearest", "zero", "slinear", "quadratic", "cubic"]
for i in style:
    function = interpolate.interp1d(x0, y, kind=i)
    y_new = function(x1)
    plb.plot(x1, y_new, label=str(i))
plb.legend(loc=3, borderaxespad=0., bbox_to_anchor=(0, 0))
plb.title('Multiple interpolation methods for one-dimensional interpolation', color='black')
plb.xlabel('x value range', color='r')
plb.ylabel('y value range', color='r')

plb.show()