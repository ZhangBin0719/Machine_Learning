'''
    作者:张斌
    时间：2019/3/31
    版本功能：使用sklearn自带的线性回归模型处理案例(无分类数据)
'''
from numpy import genfromtxt #对导入数据转化，numpy可用的矩阵
import numpy as np
from sklearn import datasets,linear_model

print("*****************第一步 导入数据 并且划分特征向量（自变量）和类别标记（因变量）*********************")
#数据导入 r转义字符，保留为字符串
deliveryData = genfromtxt(r"F:\机器学习\Machine_Learning\06MLR\Delivery.csv",delimiter=',')#分割符 ,

# print ("data：",deliveryData)

x = deliveryData[:,:-1]
y = deliveryData[:,-1]

# print (x,y)
print("*****************第二步 建立模型*********************")
lr = linear_model.LinearRegression()
lr.fit(x, y)

print (lr)

print("coefficients:",lr.coef_) #模型系数预测结果
print("intercept:",lr.intercept_)#模型截距预测结果

print("*****************第三步 预测*********************")
xPredict = [[102,6]]
yPredict = lr.predict(xPredict)
print("predict:",yPredict)
