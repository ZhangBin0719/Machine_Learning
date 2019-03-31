'''
    作者:张斌
    时间：2019/3/31
    版本功能：使用sklearn自带的线性回归模型处理案例(有分类数据,需进行转化0转化为100,1转化为001,转化为001)
    注:第一行为属性名称，注意中文，要注意编码问题
'''
from numpy import genfromtxt
from sklearn import linear_model

print("*****************第一步 导入数据 并且划分特征向量（自变量）和类别标记（因变量）*********************")
data = genfromtxt(r"F:\机器学习\Machine_Learning\06MLR\Delivery_Dummy.csv",delimiter=",",encoding='utf-8')

x = data[1:,:-1]
y = data[1:,-1]
# print (x,y)

print("*****************第二步 建立模型*********************")
mlr = linear_model.LinearRegression()
mlr.fit(x, y)

print (mlr)
# print ("coef:",mlr.coef_)#模型系数预测结果
# print ("intercept",mlr.intercept_)#模型截距预测结果

print("*****************第三步 预测*********************")
xPredict =  [[90,2,0,0,1]]
yPredict = mlr.predict(xPredict)

print ("predict:",yPredict)