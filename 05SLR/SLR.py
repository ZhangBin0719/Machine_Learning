'''
    作者:张斌
    时间:2019.3.24
    版本功能:简单线性回归的实现,为了使得建立的模型使得方差最小 从而获得回归线y=b1x+b0
'''
#简单线性回归：只有一个自变量 y=k*x+b 预测使 (y-y*)^2  最小
import numpy as np

def fitSLR(x,y):
    '''
    :param x: 自变量
    :param y: 因变量
    :return: 模型参数
    '''
    n=len(x)
    dinominator = 0
    numerator=0
    for i in range(0,n):
        numerator += (x[i]-np.mean(x))*(y[i]-np.mean(y))
        dinominator += (x[i]-np.mean(x))**2
        
    # print("numerator:"+str(numerator))
    # print("dinominator:"+str(dinominator))
    b1 = numerator/float(dinominator)
    # b0 = np.mean(y)/float(np.mean(x))
    b0=np.mean(y)-b1*np.mean(x)
    return b0,b1


# y= b0+x*b1
def prefict(x,b0,b1):
    '''
    :param x: 要预测的自变量
    :param b0:模型参数
    :param b1:模型参数
    :return:y 因变量
    '''
    return b0+x*b1
#实例
x=[1,3,2,1,3]
y=[14,24,18,17,27]

b0,b1=fitSLR(x, y)
print("intercept:{},slope:{}".format(b0,b1)) #截距 坡度
x_predict = 6
y_predict = prefict(x_predict,b0,b1)  #预测x=6
print("当自变量x取{}时,y_predict:{}".format(x_predict,y_predict))
print("简单线性回归方程为:y={}x+{}".format(b1,b0))