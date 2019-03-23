'''
    作者:张斌
    时间:2019.3.21
    版本功能:借用sklearn现有的算法实现模型建立以及预测
'''
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()
# print (iris)

knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print ("predictedLabel is :" , predictedLabel)
print (predictedLabel)