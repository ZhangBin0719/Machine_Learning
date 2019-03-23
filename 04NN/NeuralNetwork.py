'''
    作者:张斌
    时间:2019.3.23
    版本功能:BP神经网络必要类以及相关函数
'''
import numpy as np
print("****************************第一步 定义激活函数********************************")
#定义激活函数和相应的导数，双曲函数和逻辑函数
def tanh(x): #定义双曲函数
    return np.tanh(x)

def tanh_deriv(x): #定义双曲函数的导数
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x): #定义逻辑函数
    return 1/(1 + np.exp(-x))

def logistic_derivative(x): #定义逻辑函数的导数
    return logistic(x)*(1-logistic(x))
print("****************************第二步 定义神经网络的类(构造函数/建模函数/预测函数)********************************")
class NeuralNetwork:
    # 构造函数，初始化实例，，layers表示每层神经网络上的神经元个数元组，默认激活函数是双曲线函数
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        # 判断激活函数是双曲函数还是逻辑函数，初始化激活函数
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        # 初始化权重值，随机赋值
        self.weights = []
        for i in range(1, len(layers) - 1):
            # 权重的shape，是当前层和前一层的节点数目加１组成的元组  问:为何这里要多一层呢?
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)# random((layers[i - 1] + 1, layers[i] + 1))返回的是layers[i - 1] + 1行layers[i] + 1列的array每个元素的取值服从[0.0, 1.0)的均匀分布
            # print("第{}层至第{}层权重和偏好".format(i-1,i),self.weights)
            # 权重的shape，是当前层加１和后一层组成的元组
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
            # print("第{}层至第{}层权重和偏好".format(i,i+1),self.weights)

    # 创建模型，学习率置为0.2，循环系数次数10000
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X) #将x转化为矩阵形式确认x的维度至少是2维的
        temp = np.ones([X.shape[0], X.shape[1]+1]) #将矩阵置1，对bias进行初始化
        temp[:, 0:-1] = X #将x赋值给temp
        X = temp #最后一列为bias做准备 乘上随机产生的weight每行最后一个
        y = np.array(y)

        # 抽样循环
        for k in range(epochs):
            i = np.random.randint(X.shape[0]) #随机抽取一行
            a = [X[i]]#将取出的一行将其转换为列表

            # 正向更新
            for l in range(len(self.weights)):  #going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[l])))  #append()实现对应元素求和 计算输入层的输出   Computer the node value for each layer (O_i) using activation function
            error = y[i] - a[-1]  #Computer the error at the top layer 计算输出层的输出误差
            deltas = [error * self.activation_deriv(a[-1])] #For output layer, Err calculation (delta is updated error) 计算更新后的误差
            #关于为何使用导数 可看一下链接https: // blog.csdn.net / baidu_35570545 / article / details / 62065343   导数和表达式的内在关系

            # Staring backprobagation
            # 反向误差更新
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer 去掉输出层后开始__第一层(不包括第一层,即输入层)
                #Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                # 更新隐藏层的误差
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l])) #每次用最后一个(包括追加的)delta去计算,往前推进(得到一个反向的误差组)
            # 颠倒顺序
            deltas.reverse()
            # 计算权重更新
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i]) #将a转化为矩阵形式确认x的维度至少是2维的
                delta = np.atleast_2d(deltas[i]) #将deltas转化为矩阵形式确认x的维度至少是2维的
                # 权重更新
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a