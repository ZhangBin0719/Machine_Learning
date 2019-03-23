'''
    作者:张斌
    时间:2019.3.21
    版本功能:针对复杂数据irisdata.txt进行knn算法实现,不借用sklearn现有的算法
'''


import csv# csv读取数据
import random# 进行随机运算
import math
import operator

print("**********************第一步 装载数据集 并且划分训练集与测试集************************** ")
def loadDataset(filename, split, trainingSet=[], testSet=[]):
    '''
    :param filename: 装载数据集的名称(包括路径)
    :param split: 分界线，一部分做训练集，一部分做测试集
    :param trainingSet: 训练集
    :param testSet: 测试集
    :return:
    '''
    with open(filename, 'rt')as csvfile:# 装载为csvfile，逗号分隔符 读写模式:'rb'读取的是二进制文件，现在模拟读取的是文本文件使用‘rt’
        lines = csv.reader(csvfile) # 通过csv.reader读取全部行
        dataset = list(lines)# 转化成列表的数据结构赋予给dataset
        # print(dataset)# 输出dataset的值，为一个150*5二维矩阵
        # print(len(dataset))# 输出dataset的长度 =150
    for x in range(len(dataset) - 1):
        for y in range(4):
            dataset[x][y] = float(dataset[x][y])
        if random.random() < split:# 取随机值，如果随机值小于split添加到训练集
            trainingSet.append(dataset[x])
        else:
            testSet.append(dataset[x])
    # print(trainingSet) # 打印出训练集
    # print(testSet) # 打印出训练集

print("**********************第二步 计算两个样本欧式距离************************** ")
def euclideanDistance(instance1, instance2, length):
    '''
    :param instance1: 第一个实例向量
    :param instance2: 第二个实例向量
    :param length: 数据维度 为测试集的个数-1 最后一个为类别标记
    :return:返回两个样本间的距离
    '''
    distance = 0
    # 对于每一维计算实例之间的差，并对其进行平方
    for x in range(length):
        distance += pow(instance1[x] - instance2[x], 2)
    return math.sqrt(distance)

print("**********************第三步 返回离测试实例最近的K个训练集样本 并对距离进行排序************************** ")
def getNeighbors(trainingSet, testInstance, k):
    distances = []# 创建一个容器，装所有的距离
    length = len(testInstance) - 1# 定义一个长度，测试的实例的维度减一 最后一个为类别标记
    for x in range(len(trainingSet)):# 对于训练集中的每一个样本计算到测试集实例的距离
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))# 将计算出的欧式距离dist连同trainingSet[x]一起存储到distances[]
    # print(len(distances))# 输出distances的长度即为划分为训练集的数据个数

    # 按从小到大对distances进行排序返回到distances
    distances.sort(key=operator.itemgetter(1))# operator模块提供的itemgetter函数用于获取对象的哪些维的数据,此处取得是[1]处的值
    # print(distances)# 输出排序后的distances

    neighbors = []# 定义一个neighbor[]存储最近的邻距点

    for x in range(k):# k为设置取几个邻近点的值，将最近的k个点存储到neighbors[]中
        neighbors.append(distances[x][0]) # 将最近的k个点中的trainingSet[] 保存到neighbors中,即没有保存距离数值
    # print(neighbors)# 输出最近的k个点
    return neighbors

print("**********************第四步 根据最近的邻距，进行投票，少数服从多数，预测实例归类************************** ")
def getResponse(neighbors):
    classVotes = {}# 定义一个classVotes的字典

    for x in range(len(neighbors)):
        response = neighbors[x][-1]# 取neighbors的最后一个值赋给response,即属于哪一类
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    # print(classVotes[response]) # 输出投票数
    # print(classVotes.items()) # 输出classVotes.items()即分类结果和投票数

    # 对于每一个类的投票按降序的方式排序
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    '''
        参数:classVotes.items()为对象
             operator.itemgetter函数，指定对对象第几个域进行排序，域对应的取值从0开始
             reverse参数是一个bool变量，默认为false（升序排列），定义为True时将按降序排列。
    '''
    # print(sortedVotes)    # 输出与classVotes.items()一样
    # print(sortedVotes[0][0])# 输出sortedVotes的[0]值即名称
    # print(sortedVotes[0][1]) # 输出sortedVotes的[1]值即投票值
    return sortedVotes[0][0] # 返回投票最多的类别

print("**********************第五步 预测准确率 评估************************** ")
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]: # [-1]取最后一个值即lable
            correct += 1
    return (correct / float(len(testSet))) * 100


def main():
    trainingSet = []# 创建两个空的训练集和测试集
    testSet = []
    split = 0.67# 从0-1中的分布中，调用loadDtaset把三分之二划分为训练集，三分之一划分为测试集。
    loadDataset(r'F:\机器学习\Machine_Learning\02KNN\irisdata.txt', split, trainingSet, testSet)# 第一参数：传入路径r 开头的python字符串是 raw 字符串，所以里面的所有字符都不会被转义
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    predictions = [] # 定义predictions数据集存储预测出类的值
    k = 3# 取最近的k个邻距
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted=' + repr(result) + ',actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()
