# -*- coding: utf-8 -*-
#感知器
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Perceptron:
    """感知机分类器

    参数
    ------------
    eta : float, 默认 0.01
        学习率 (范围 0.0 ~ 1.0)
    n_iter : int, 默认 50
        训练数据集迭代次数
    random_state : int, 默认 1
        随机数种子，用于权重初始化

    属性
    -----------
    w_ : 一维数组
        训练后的权重向量（包含偏置项）
    errors_ : 列表
        每次迭代的误分类样本数
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """初始化感知机参数"""
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """训练模型

        参数
        ----------
        X : 二维数组，形状 [样本数, 特征数]
            训练数据矩阵
        y : 一维数组，形状 [样本数]
            目标分类标签（-1 或 1）

        返回
        -------
        self : 对象
            训练后的模型实例
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # 权重迭代更新
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 计算权重更新量
                update = self.eta * (target - self.predict(xi))
                # 更新权重（包含偏置项）
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """计算净输入 z = w·x + b"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """返回类别预测结果（单位阶跃函数）"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
#鸢尾花
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', s)

df = pd.read_csv(s,
                 header=None,
                 encoding='utf-8')

print(df.tail())

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()

ppn=Perceptron(eta=0.1,n_iter=10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

#自适应线性神经元
class AdalineGD(object):
    """自适应线性神经元（ADALINE）分类器（使用梯度下降法）
    
    该分类器通过梯度下降法最小化平方和代价函数，实现Adaline算法。

    Parameters
    ----------
    eta : float, 默认值=0.01
        学习率（通常介于0.0到1.0之间），控制梯度下降的步长
    n_iter : int, 默认值=50
        对训练数据的遍历次数（迭代轮数）
    random_state : int, 默认值=1
        用于权重初始化的随机数生成器种子

    Attributes
    ----------
    w_ : ndarray of shape (1 + n_features,)
        训练后的权重向量，其中w_[0]表示偏置项
    cost_ : list
        每次迭代中所有样本的平方和代价函数平均值

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> ada = AdalineGD(eta=0.0001, n_iter=100)
    >>> ada.fit(X, y)
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """初始化AdalineGD参数"""
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """使用梯度下降法拟合训练数据

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            训练数据矩阵，每行代表一个样本
        y : array-like, shape (n_samples,)
            目标值（类别标签），整数类型

        Returns
        -------
        self : object
            返回实例自身以支持链式调用

        Notes
        -----
        权重更新遵循梯度下降规则：
            w := w + η * Σ_i (y_i - φ(z_i)) * x_i
        其中φ为激活函数（Adaline中使用恒等函数）
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """计算净输入 z = w·x + b
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            输入数据矩阵

        Returns
        -------
        z : ndarray, shape (n_samples,)
            权重与输入的线性组合加上偏置项
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """线性激活函数（恒等函数）
        
        注意：此方法为保持与其他分类器的接口一致性而保留，
        实际对于Adaline算法，激活函数等价于净输入
        """
        return X

    def predict(self, X):
        """使用单位阶跃函数预测类别标签
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            待分类的输入数据

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            预测的类别标签（-1 或 1）
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
    def plot_decision_regions(X, y, classifier, resolution=0.02):
        """可视化二维特征空间的决策区域
        
        参数
        ----------
        X : array-like, 形状 (n_samples, 2)
            二维特征矩阵，仅支持两个特征的可视化
        y : array-like, 形状 (n_samples,)
            样本对应的类别标签
        classifier : 分类器对象
            实现predict方法的分类器实例
        resolution : float, 可选（默认=0.02）
            网格点分辨率，控制决策边界的平滑度

        返回
        -------
        None : 无返回值
            直接显示可视化图形

        实现说明
        ----------
        1. 创建坐标网格：生成覆盖特征范围的网格点矩阵
        2. 预测分类结果：使用分类器预测网格所有点的分类
        3. 绘制决策区域：通过等高线填充不同类别的区域
        4. 绘制样本点：用不同标记展示实际数据分布
        """
        # 初始化图形标记和配色方案
        markers = ('s', 'x', 'o', '^', 'v')  # 方形、叉号、圆圈、三角、倒三角
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # 颜色序列
        cmap = ListedColormap(colors[:len(np.unique(y))])  # 根据类别数量创建色板

        # 生成决策面网格数据
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 特征1范围
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 特征2范围
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),  # 特征1网格
            np.arange(x2_min, x2_max, resolution)   # 特征2网格
        )
        
        # 预测网格点类别并绘制决策区域
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)  # 调整形状匹配网格
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)  # 半透明填充决策区域
        plt.xlim(xx1.min(), xx1.max())  # 设置x轴范围
        plt.ylim(xx2.min(), xx2.max())  # 设置y轴范围

        # 绘制各类别样本点
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(
                x=X[y == cl, 0],        # 当前类别的特征1值
                y=X[y == cl, 1],        # 当前类别的特征2值
                alpha=0.8,              # 点透明度
                c=colors[idx],          # 对应颜色
                marker=markers[idx],    # 形状标记
                label=cl,               # 图例标签
                edgecolor='black'       # 点边缘颜色
            )

#鸢尾花2       
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_gd = AdalineGD(n_iter=15, eta=0.01)
ada_gd.fit(X_std, y)

AdalineGD.plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/02_14_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('images/02_14_2.png', dpi=300)
plt.show()



class AdalineSGD(object):
    """自适应线性神经元分类器（随机梯度下降法实现）
    
    该分类器通过随机梯度下降法（SGD）实现Adaline算法，适用于大规模数据集。
    每次迭代随机选取样本更新权重，支持在线学习和增量式训练。

    Parameters
    ------------
    eta : float, 默认值=0.01
        学习率（通常介于0.0到1.0之间），控制参数更新的步长
    n_iter : int, 默认值=10
        训练数据的遍历次数（迭代轮数）
    shuffle : bool, 默认值=True
        若为True，每次迭代前打乱训练数据顺序，防止周期震荡
    random_state : int, 默认值=None
        随机数生成器种子，用于可重复的权重初始化

    Attributes
    -----------
    w_ : 一维数组
        训练后的权重向量（包含偏置项）
    cost_ : 列表
        每次迭代的平均损失值（使用平方误差计算）

    方法
    -------
    fit(X, y) : 
        主训练方法，执行多轮迭代训练
    partial_fit(X, y) : 
        增量式训练方法，不重新初始化权重
    predict(X) : 
        返回样本的预测类别标签
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        """初始化SGD参数"""
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False  # 权重初始化状态标记
        self.shuffle = shuffle      # 是否打乱数据标志
        self.random_state = random_state
        
    def fit(self, X, y):
        """执行模型训练
        
        Parameters
        ----------
        X : 二维数组，形状 [样本数, 特征数]
            训练数据矩阵
        y : 一维数组，形状 [样本数]
            目标分类标签（-1 或 1）

        Returns
        -------
        self : 对象
            返回训练后的模型实例

        实现流程：
        1. 初始化权重
        2. 进行n_iter轮迭代：
            a. 可选打乱数据顺序
            b. 逐个样本更新权重
            c. 计算并记录平均损失
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)  # 计算本轮平均损失
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """增量式训练（适合在线学习场景）
        
        特点：
        - 不重新初始化权重
        - 支持单样本或小批量数据训练
        - 适用于实时数据流场景
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """内部方法：打乱训练数据顺序
        
        使用numpy的permutation函数生成随机索引
        返回打乱顺序后的X和y
        """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """内部方法：权重初始化
        
        使用正态分布初始化权重向量：
        - loc=0.0：均值设为0
        - scale=0.01：标准差设为0.01
        - 权重向量维度：特征数+1（含偏置项）
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """内部方法：单样本权重更新
        
        实现步骤：
        1. 计算净输入
        2. 计算预测误差
        3. 更新权重向量
        4. 返回当前样本的损失值
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)  # 更新特征权重
        self.w_[0] += self.eta * error           # 更新偏置项
        cost = 0.5 * error**2                    # 计算单样本损失
        return cost
    
    def net_input(self, X):
        """计算净输入 z = w·x + b"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """线性激活函数（保持接口统一）"""
        return X

    def predict(self, X):
        """预测类别标签（单位阶跃函数）
        
        返回
        -------
        y_pred : 数组，元素值为1或-1
            预测的类别标签
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)