from sklearn.feature_extraction import DictVectorizer  #数据格式 类型转换整形
import csv  #导入数据
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
#rb修改为rt
'''
    文件操作:
    步骤:打开文件->操作文件(读写等)->关闭文件
    打开文件:open(filename文件名(包括路径),mode(打开模式))打开模式:r只读,文件不存在报错/w只写,文件不存在自动创建/a文件末尾追加/r+ 读写
    操作文件:写入 写入write()文本数据写入  writelines()字符串列表写入
            读取  read()包含整个文件内容的一个字符串 readline()文件下一行内容字符串 readlines() 整个文件内容的列表,每项是以换行符为结尾的一行字符串
    关闭文件:close()
    注:路径前的'r'是防止字符转义的 如果路径中出现'\t'的话 不加r的话\t就会被转义 而加了'r'之后'\t'就能保留原有的样子
     整体方式:with open(filename文件名(包括路径),mode(打开模式)) as f:
'''
#第一步文件初处理
#打开文件 创建读写器
allElectronicsData = open(r'/机器学习/Machine_Learning/01DTree/AllElectronics.csv', 'rt')
reader = csv.reader(allElectronicsData)
#headers = reader.next()   #python2中用法  下面为python3的用法
#打印csv文件中的第一行标题header
headers = next(reader)
print(headers)

featureList = []  #存放特征值
labelList = []    #存放分类结果
#循环遍历csv文件每行
for row in reader:
    #将分类结果存入labelList
    labelList.append(row[len(row)-1])
    rowDict = {}
    #循环遍历csv文件每行内的每项内容
    for i in range(1, len(row)-1):
        #键值对匹配 属性名和值匹配
        rowDict[headers[i]] = row[i]
    #字典存入featureList
    featureList.append(rowDict)

print(featureList)

#第二步 特征值的矩阵转化
# Vetorize features
#实例化vec,进行特征值的转化,vec.fit_transform转化为0-1矩阵
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()
print("dummyX: " + str(dummyX))

#vec.get_feature_names获取属性名与类别匹配
print(vec.get_feature_names())
print("labelList: " + str(labelList))

# vectorize class labels
#实例化lb,进行分类结果的转化,lb.fit_transform转化为0-1矩阵
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))

#第三步 决策树的建立
# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
#决策树的简历 决策树信息熵的选择用ID3,此外还gini函数
clf = tree.DecisionTreeClassifier(criterion='entropy')
#分类器的学习
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

#可视化决策树
# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
    
#第四步 预测
#修改数据后进行预测
#注意维度 这里是[[数组元素]] 用reshape  1代表1行,-1代表未知,在这里代表10
oneRowX = dummyX[0, :].reshape(1, -1)
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0][0] = 1
newRowX[0][2] = 0
newRowX.reshape(1, -1)
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY: " + str(predictedY))