'''
    作者：张斌
    时间：2019/3/31
    版本功能：实现运用随机数据去建立非线性回归模型
'''
import numpy as np
import random

#创建数据
def genData(numPoints,bias,variance):
    '''
    :param numPoints: 行数
    :param bias: 对y所加的偏好
    :param variance: 方差
    :return: 特征向量/类别标记
    '''
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=(numPoints))
    for i in range(0,numPoints):
        x[i][0]=1
        x[i][1]=i
        y[i]=(i+bias)+random.uniform(0,1)*variance
    return x,y

#创建梯度下降法求解theta 通用 不需要归一、也不需要二值转化
def gradientDescent(x,y,theta,alpha,m,numIterations):
    '''
    :param x: 特征向量
    :param y: 分类标记
    :param theta:要求的向量值 系数值
    :param alpha:学习率
    :param m:实例数
    :param numIterations:更新的次数
    :return:theta
    '''
    xTran = np.transpose(x)#矩阵转置
    for i in range(numIterations):
        hypothesis = np.dot(x,theta)#做内积 z=theta0*x0+theta1*x1+...+thetan*xn
        loss = hypothesis-y#预测值-标记值的差h(theta)x(i)-y(i)
        cost = np.sum(loss**2)/(2*m) #此处只定义最简单的cost函数，没有照资料的
        gradient=np.dot(xTran,loss)/m
        theta = theta-alpha*gradient#更新法则
        # print ("Iteration :%d | cost :%f" %(i,cost))
    return theta

x,y = genData(100, 25, 10)#生成数据 特征向量：100*2 类别标记：100*1 偏好：25 方差：10
# print ("x:",x)
# print ("y:",y)

m,n = np.shape(x)
m_y = np.shape(y)
# print("m:"+str(m)+" n:"+str(n)+" m_y:"+str(m_y))

numIterations = 100000  #循环次数
alpha = 0.0005       #学习率，一般0-1之间 之后可设置为自动改近（由大到小）
theta = np.ones(n)   #参数
theta= gradientDescent(x, y, theta, alpha, m, numIterations)
print("参数为：",theta)