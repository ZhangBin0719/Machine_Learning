'''
    作者:张斌
    时间:2019.3.21
    版本功能:复杂数据实现svm算法 核函数:高斯(径向基)函数 人脸识别
'''
# 在python2.x版本中要使用Python3.x的特性,可以使用__future__模块导入相应的接口，减少对当前低版本影响
# from __future__ import print_function

from time import time# 计时，程序运行时间
import logging # 打印程序进展时的一些信息
import matplotlib.pyplot as plt# 最后识别出来的人脸通过绘图打印出来
# 当import 一个模块比如如下模块cross_validation时，会有删除横线，表示该模块在当前版本可能已经被删除,在新版本中改为model_selection模块
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# 导入混淆矩阵模块confusion_matrix()
from sklearn.metrics import confusion_matrix

# print(__doc__) 注释导出
'''
logging.basicConfig函数各参数:logging.basicConfig(level,forma,datefmt,filename,filemode)
    level: 设置日志级别，默认为logging.WARNING
    filename: 指定日志文件名 filemode: 和file函数意义相同，指定日志文件的打开模式，'w'或'a'
    format: 指定输出的格式和内容，format可以输出很多有用信息，如上例所示: 
         %(levelno)s: 打印日志级别的数值
         %(levelname)s: 打印日志级别名称
         %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
         %(filename)s: 打印当前执行程序名
         %(funcName)s: 打印日志的当前函数
         %(lineno)d: 打印日志的当前行号
         %(asctime)s: 打印日志的时间
         %(thread)d: 打印线程ID
         %(threadName)s: 打印线程名称
         %(process)d: 打印进程ID
         %(message)s: 打印日志信息
        datefmt: 指定时间格式，同time.strftime()
        stream: 指定将日志的输出流，可以指定输出到sys.stderr,sys.stdout或者文件，默认输出到sys.stderr，当stream和filename同时指定时，stream被忽略
'''
print("*************************第一步 数据下载 数据预处理*************************")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') #程序进展打印
'''
    fetch_lfw_people(data_home=None, funneled=True, resize=0.5, 
    min_faces_per_person=0, color=False, slice_=(slice(70, 195, None), 
    slice(78, 172, None)), download_if_missing=True, return_X_y=False)
'''
# Download the data, if not already on disk and load it as numpy arrays 数据集导入,如果每个数据集则下载
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)  #导入名人人脸数据集(类似字典)
# print(lfw_people)
# introspect the images arrays to find the shapes (for plotting) 观察照片去发现形状特征(为了做图的方便)
n_samples, h, w = lfw_people.images.shape  # 多少个实例，h,w高度，宽度值

# for machine learning we use the 2 data directly (as relative pixel positions info is ignored by this model)
X = lfw_people.data # 特征向量矩阵 每一行是个实例，每一列是个特征值
# print("特征向量矩阵",X)
n_features = X.shape[1] # n_featers表示的就是维度 返回行/列,也就是每个人对应会提取多少的特征值
# print("每个人提取{}个特征值".format(n_features))

y = lfw_people.target # 返回每一组的特征标记 即不同的人的身份  返回数字1\2\3\4\5\6...
# print("y : ",y )
target_names = lfw_people.target_names #人名 返回人名
# print("target_names: ",target_names)
#shape函数以元组形式返回数组各个维度的元素个数
n_classes = target_names.shape[0]#人数 shape返回行

print("Total dataset size:")
print("n_samples: ",n_samples) #实例个数
print("n_features: ",n_features)#特征向量维度
print("n_classes: ",n_classes)#总共类数


print("******************第二步 数据集拆分*******************")
# 将训练集拆分四部分(特征向量的训练与测试集/类别标记的训练与测试集)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


print("******************第三步 数据集降维 建立主成分分析(PCA)模型*******************")
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeleddataset): unsupervised feature extraction / dimensionality reduction
# PCA降维方法，减少特征值，降低复杂度。
n_components = 150# PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
print("Extracting the top {} eigenfaces from {} faces".format(n_components, X_train.shape[0]))

t0 = time()          #返回当前时间的时间戳
'''
    svd_solver：即指定奇异值分解SVD的方法,有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。
                randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 
                full则是传统意义上的SVD，使用了scipy库对应的实现。
                arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用
                了scipy库的sparse SVD实现。
    n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目
    whiten ：判断是否进行白化，就是对降维后的数据的每个特征进行归一化白化，让方差都为1.对于PCA降维本身来说，一般不需要
             白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。
    components_ ：特征空间中的主轴，表示数据中最大方差的方向。按explain_variance_排序。
    两个PCA类的成员值得关注:explained_variance_，它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
                          explained_variance_ratio_，它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。
'''
pca = PCA(svd_solver='randomized', n_components=n_components, whiten=True).fit(X_train)  # 训练一个pca模型
print("Train PCA in %0.3fs" % (time() - t0)) #训练模型花的时间

eigenfaces = pca.components_.reshape((n_components, h, w))  # 提取人脸上的特征值eigenfaces

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time() #返回当前时间的时间戳
# 将训练集与测试集降维
X_train_pca = pca.transform(X_train) # 特征量X中训练集所有的特征向量通过pca转换成更低维的矩阵
X_test_pca = pca.transform(X_test) # 特征量X中测试集所有的特征向量通过pca转换成更低维的矩阵
print("Done PCA in %0.3fs" % (time() - t0)) #降维花的时间


print("******************第三步 训练SVM分类器*******************")
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time() #返回当前时间的时间戳
# param_grid把参数设置成了不同的值，C：权重；gamma：多少的特征点将被使用，因为我们不知道多少特征点最好，选择了不同的组合
'''关于高斯核函数 https://blog.csdn.net/lin_limin/article/details/81135754   γ=−1/2σ^2
    ① 高斯核的参数σ取值与样本的划分精细程度有关：σ越小，低维空间中选择的曲线越复杂，试图将每个样本与其他样本都区分开来；分的类别越细，容易出现过拟合；σσ越大，分的类别越粗，可能导致无法将数据区分开来，容易出现欠拟合。 
　　② 惩罚因子C的取值权衡了经验风险和结构风险：C越大，经验风险越小，结构风险越大，容易出现过拟合；C越小，模型复杂度越低，容易出现欠拟合。
'''
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],  # C是对错误的惩罚 惩罚因子
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], } #gamma 代表多少比例特征点被使用
# 把所有我们所列参数的组合(30种)都放在SVC里面进行 计算，最后看出哪一组函数的表现度最好
'''
    GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, \
                 n_jobs=1, iid=True, refit=True, cv=None, \
                 verbose=0, pre_dispatch=‘2*n_jobs’, error_score=’raise’, \
                 return_train_score=’warn’)
    GridSearchCV参数解释:estimator:所使用的基础模型，比如svc
                         param_grid:是所需要的调整的参数，以字典或列表的形式表示
                         scoring:准确率评判标准
                         n_jobs:并行运算数量（核的数量 ），默认为1，如果设置为-1，则表示将电脑中的cpu全部用上
                         iid:假设数据在每个cv(折叠)中是相同分布的，损失最小化是每个样本的总损失，而不是折叠中的平均损失。
                         refit:默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。
                         cv:交叉验证折叠数，默认是3，当estimator是分类器时默认使用StratifiedKFold交叉方法，其他问题则默认使用KFold
                         verbose:日志冗长度，int类型，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出
                         pre_dispatch:控制job数量，避免job过多出现内存错误
    GridSearchCV对象:cv_results_:用来输出cv结果的，可以是字典形式也可以是numpy形式，还可以转换成DataFrame格式
                     best_estimator_：通过搜索参数得到的最好的估计器，当参数refit=False时该对象不可用
                     best_score_：float类型，输出最好的成绩
                      best_params_:通过网格搜索得到的score最好对应的参数
    GridSearchCV方法:decision_function(X):返回决策函数值（比如svm中的决策距离）
                    predict_proba(X):返回每个类别的概率值（有几类就返回几列值）
                    predict(X)：返回预测结果值（0/1）
                    score(X, y=None):返回函数
                    get_params(deep=True):返回估计器的参数
                    fit(X,y=None,groups=None,fit_params)：在数据集上运行所有的参数组合
                    transform(X):在X上使用训练好的参数        
'''
clf = GridSearchCV(SVC(kernel='rbf'), param_grid) #采用高斯核函数：K(x,z)=exp(−γ||x−z||2)K(x,z)=exp(−γ||x−z||2) rbf/poly/sigmold
clf = clf.fit(X_train_pca, y_train)#找到边际最大超平面
print("Done Fiting in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


print("******************第四步 SVM模型评估*******************")
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("Done Predict in %0.3fs" % (time() - t0))
'''
    def classification_report(y_true, y_pred, labels=None, target_names=None,
                              sample_weight=None, digits=2, output_dict=False)
    函数用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。 
    主要参数: 
        y_true：1维数组，或标签指示器数组/稀疏矩阵，目标值。 
        y_pred：1维数组，或标签指示器数组/稀疏矩阵，分类器返回的估计值。 
        labels：array，shape = [n_labels]，报表中包含的标签索引的可选列表。 
        target_names：字符串列表，与标签匹配的可选显示名称（相同顺序）。 
        sample_weight：类似于shape = [n_samples]的数组，可选项，样本权重。 
        digits：int，输出浮点值的位数．
'''
print(classification_report(y_test, y_pred, target_names=target_names))
'''
    混淆矩阵是一个误差矩阵, 常用来可视化地评估监督学习算法的性能
    def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    混淆矩阵 C 中的元素 Ci,j 等于真实值为组 i , 而预测为组 j 的观测数(the number of observations). 
    所以对于二分类任务, 预测结果中, 正确的负例数(true negatives, TN)为 C0,0; 错误的负例数(false negatives, FN)为 C1,0; 
    真实的正例数为 C1,1; 错误的正例数为 C0,1.
    对角线上的代表预测与真实值一样的
'''
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# 利用Matplotlib对预测进行定性评估
'''
    调整子图布局
    subplots_adjust(self, left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=None)
        参数含义（和建议的默认值）是：left  = 0.125  #图片中子图的左侧
        right = 0.9# 图片中子图的右侧
        bottom = 0.1# 图片中子图的底部
        top = 0.9 # 图片中子图的顶部
        wspace = 0.2 #为子图之间的空间保留的宽度，平均轴宽的一部分
        hspace = 0.2 #为子图之间的空间保留的高度，平均轴高度的一部分
'''
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row)) # 在figure上建立一个图当背景
    #布局
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)#plt.subplot(a, b, c)  a：代表子图的行数 b：代表该行图像的列数 c:代表每行的第几个图像
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)  #imshow(I,[low high])用指定的灰度范围 [low high]显示灰度图像I。
        plt.title(titles[i], size=12)
        plt.xticks(()) #xticks()返回了两个对象,一个是刻标(locs)，另一个是刻度标签  locs, labels = xticks()
        plt.yticks(())


# 把预测的函数归类标签和实际函数归类标签
'''
    split(分隔符，分割几次)从左向右寻找，以某个元素为中心将左右分割成两个元素并放入列表中，该分隔符丢弃
    rsplit(分隔符，分割几次)从右向左寻找，以某个元素为中心将左右分割成两个元素并放入列表中，该分隔符丢弃
    splitlines(分隔符，分割几次)根据换行符（\n）分割并将元素放入列表中，该分隔符丢弃

'''
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
print("prediction_titles ",prediction_titles )
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces
# 降维后图片
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
# 提取过特征向量之后的特征脸
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()