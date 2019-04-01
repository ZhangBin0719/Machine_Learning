'''
    作者：张斌
    时间：2019/4/1
    版本功能：回归模型建立中一些相关度和决定系数的计算
'''
import numpy as np
import math

#构建函数计算相关度
def computeCorrelation(X,Y):
    '''
    :param X:向量X
    :param Y:向量Y
    :return:相关度
    '''
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0,len(X)):
        diffXXBar = X[i]-xBar
        diffYYBar = Y[i]-yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    return SSR/SST

def polyfit(x,y,degree):
    '''
    :param x:自变量
    :param y:因变量
    :param degree:回归的最高次数
    :return:
    '''
    results = {}
    coeffs = np.polyfit(x,y,degree) #建立回归模型，此处degree为1,返回系数

    results['polynomial'] = coeffs.tolist()#返回值转化为list

    p = np.poly1d(coeffs) #一维模型函数

    yhat = p(x)#y的预测值
    ybar = np.sum(y)/len(y)#均值
    ssreg = np.sum((yhat-ybar)**2)
    print("ssreg:",ssreg)
    sstot = np.sum((y-ybar)**2)
    print("sstot:", sstot)
    results['determination'] = ssreg / sstot

    return results

testX = [1,3,8,7,9]
testY = [10,12,24,21,34]

print ("皮尔逊相关度（r）为：",computeCorrelation(testX,testY))
print ("r^2为：",computeCorrelation(testX,testY) ** 2)
print("R平方值：",polyfit(testX,testY,1)['determination'])