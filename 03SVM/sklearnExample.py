'''
    作者:张斌
    时间:2019.3.21
    版本功能:复杂数据实现svm算法,并画出超平面
'''
'''
    svm算法相关属性:
    （1）support_ : 是一个array类型，它指的是训练出的分类模型的支持向量的索引，即在所有的训练样本中，哪些样本成为了支持向量。
    （2）support_vectors_: 支持向量的集合，即汇总了当前模型的所有的支持向量。
    （3）n_support_ : 比如SVC将数据集分成了4类，该属性表示了每一类的支持向量的个数。
    （4）dual_coef_ :array, shape = [n_class-1, n_SV]对偶系数支持向量在决策函数中的系数，在多分类问题中，这个会有所不同。
    （5）coef_ : array,shape = [n_class-1, n_features]该参数仅在线性核时才有效，指的是每一个属性被分配的权值
    （6）intercept_ :array, shape = [n_class * (n_class-1) / 2]决策函数中的常数项bias。和coef_共同构成决策函数的参数值。
'''
import numpy as np  #数学算法
import pylab as pl  #画图
from sklearn import svm #svm算法

# we create 40 separable points
'''
np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
'''
#第一步 输入数据
np.random.seed(0) #固定方法,每次随机值保持不变
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]] #训练实例,randn(20,2)正态分布生成20个2维的 -[2,2]和+[2,2]为了区分左右
Y = [0]*20 +[1]*20  #分类标记
#观察特征和分类结果
# print("X:",X)
# print("Y:",Y)

#fit the model 第二部 建模
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

#第三步 画图
#得到超平面
# get the separating hyperplane
# switching to the generic n-dimensional parameterization of the hyperplan to the 2D-specific equation
# of a line y=a.x +b: the generic w_0x + w_1y +w_3=0 can be rewritten y = -(w_0/w_1) x + (w_3/w_1)
w = clf.coef_[0]
print ("w: ", w)
a = -w[0]/w[1]
print ("a: ", a)
xx = np.linspace(-5, 5)
yy = a*xx - (clf.intercept_[0])/w[1]
print ("support_vectors_: ", clf.support_vectors_)
print ("clf.coef_: ", clf.coef_)
# print ("xx: ", xx)
# print ("yy: ", yy)

# plot the parallels to the separating hyperplane that pass through the support vectors
b = clf.support_vectors_[0] #得到第一个支持向量
yy_down = a*xx + (b[1] - a*b[0])
b = clf.support_vectors_[-1] #得到最后一个支持向量
yy_up = a*xx + (b[1] - a*b[0])


# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')
'''
    详见:https://blog.csdn.net/weixin_40713373/article/details/80024583
    函数原型pl.scatter(x,y,s=20,c='b',maker='o',cmpa=None,norm=None,vmin=None,vax=None,alpha=None,linewidths=None,verts=None,hole=None)
    参数介绍:(1)x,y是相同长度的数组
            (2)s可以是标量，或者与x,y长度相同的数组，表明散点的大小。默认20
            (3)c，即color，是点的颜色。颜色参数如下：b-blue   c-cyan  g-greeen  k-black  m-magenta  r-red  w-white  y-yellow
            (4)marker 是散点的形状。其属性较多，. --点  o--圆圈  ，--像素  v--倒三角  *--星星
'''
pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=80,c='b')
'''
    plt.cm中cm全称表示colormap，paired表示两个两个相近色彩输出，比如浅蓝、深蓝；浅红、深红；浅绿，深绿这种。
'''
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()