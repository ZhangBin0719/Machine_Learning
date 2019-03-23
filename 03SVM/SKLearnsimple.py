'''
    作者:张斌
    时间:2019.3.21
    版本功能:通过sklearn中已有的svm算法进行实现,通过简单数据[[2, 0], [1, 1], [2, 3]]
'''
from sklearn import svm

x = [[2, 0], [1, 1], [2, 3]]  #实例点
y = [0, 0, 1]                 #分类标记
clf = svm.SVC(kernel = 'linear')  #分类器 核函数采用线性
#建立模型
clf.fit(x, y)

print (clf)

# get support vectors
print (clf.support_vectors_)
# get indices of support vectors
print (clf.support_)
# get number of support vectors for each class
print (clf.n_support_)
# predict  在这里加入[[2,0]]时注意,要加入列表,而不是一个数组
print(clf.predict([[2,0],[1,0],[3,0]]))