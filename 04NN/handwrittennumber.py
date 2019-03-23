'''
    作者:张斌
    时间:2019.3.23
    版本功能:BP神经网络实现主程序
'''

# 每个图片8x8  识别数字：0,1,2,3,4,5,6,7,8,9

import numpy as np
import pylab as pl #查看图像
from sklearn.datasets import load_digits #手写字体数据集
from sklearn.metrics import confusion_matrix, classification_report #混淆矩阵/分类报告(精确率 召回率 F1) 对结果的衡量
from sklearn.preprocessing import LabelBinarizer #将[0,9]转化为二维数字类型 0=0000000000 1=1000000000 2=0100000000 3=0010000000 ... 是几,第几位就是1
from NeuralNetwork import NeuralNetwork
from sklearn.model_selection import train_test_split#划分训练集与测试集

print("**************************BP神经网络应用:手写字体识别**********************")
print("**************************第一步 数据集的导入**********************")
digits = load_digits() #导入手写字体数据集
# #图像的查看
# print("数据的维度:",digits.data.shape)
# pl.gray()#转化为灰度图片
# pl.matshow(digits.images[0])#下标可以0-9
# pl.show()

X = digits.data
print("特征向量:",X)
y = digits.target
print("类别标记:",y)
X -= X.min()  # 规范化值，使其进入范围0-1
X /= X.max()  # 规范化值，使其进入范围0-1

nn = NeuralNetwork([64, 100, 10], 'logistic') #实例化一个神经网络  层数设定:可调整 而维数:由输入数据和输出数据决定 隐藏层较灵活,一般比输入层大
X_train, X_test, y_train, y_test = train_test_split(X, y) #数据集划分训练集和测试集
labels_train = LabelBinarizer().fit_transform(y_train)#类别标记转化
labels_test = LabelBinarizer().fit_transform(y_test)
print ("**************************************start fitting**************************")
nn.fit(X_train, labels_train, epochs=3000)
predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o)) #numpy.argmax(a, axis=None, out=None) 返回沿轴axis最大值的索引。 即最大的概率对应的那个数
print ("混淆矩阵为:(主对角线上数字越大越好)\n",confusion_matrix(y_test, predictions))
print ("分类报告:精确率 召回率 F1值\n",classification_report(y_test, predictions))