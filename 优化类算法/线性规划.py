# -*- encoding: utf-8 -*-
"""
@Modify Time : 2024/4/4 15:57      
@Author : Mozinapig   
@Version : 1.0  
@Desciption : None
  
"""
from scipy import optimize
import numpy as np
c = np.array([2,3,-5])
A = np.array([[-2,5,-1],[1,3,1]])
B = np.array([-10,12])
Aeq = np.array([[1,1,1]])
Beq = np.array([7])
x1 = [0, None]  #未知数取值范围
x2 = [0, None]  #未知数取值范围
x3 = [0, None]  #未知数取值范围
res = optimize.linprog(-c,A,B,Aeq,Beq,bounds=(x1, x2, x3))
print(res)