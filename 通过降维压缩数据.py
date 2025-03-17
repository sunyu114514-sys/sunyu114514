# coding: utf-8


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from packaging.version import parse as version
from scipy import __version__ as scipy_version
from numpy import exp
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
import matplotlib
from sklearn.multiclass import OneVsRestClassifier
matplotlib.use('TkAgg')
#解决负号问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载葡萄酒数据集（UCI机器学习仓库）
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)  # 数据集没有表头

# 分割特征和标签
# 第0列为类别标签（葡萄酒品种），第1列开始为13个化学特征
X, y = df_wine.iloc[:, 1:].to_numpy(), df_wine.iloc[:, 0].values
# 将数据集分割为训练集（70%）和测试集（30%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)  # 固定随机种子保证可重复
# 数据标准化处理（Z-score标准化）
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)  # 训练集拟合并转换
X_test_std = sc.transform(X_test)        # 测试集使用相同参数转换

# 计算协方差矩阵（特征维度间的协方差）
cov_mat = np.cov(X_train_std.T)
# 特征值分解（计算主成分）
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# 计算方差解释率
tot = sum(eigen_vals)  # 总方差
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]  # 单个主成分方差占比
cum_var_exp = np.cumsum(var_exp)  # 累计方差占比
# 绘制方差解释率可视化图表
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')  # 各主成分方差贡献
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='Cumulative explained variance')  # 累计方差贡献
plt.ylabel('Explained variance ratio')  # Y轴标签
plt.xlabel('Principal component index')  # X轴标签
plt.legend(loc='best')  # 图例位置
plt.tight_layout()  # 自动调整布局
# plt.savefig('images/05_02.png', dpi=300)  # 保存图片（可选）
plt.show()  # 显示图表


# 生成特征值-特征向量配对列表（取特征值绝对值确保正数）
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
# 按特征值降序排序（保留最大方差方向）
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# 构建投影矩阵（取前两个最大特征值对应的特征向量）
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],  # 第一主成分方向
               eigen_pairs[1][1][:, np.newaxis])) # 第二主成分方向
# 将单个样本投影到主成分空间（示例）
X_train_std[0].dot(w)  # 得到该样本在二维空间的坐标
# 将所有训练数据投影到主成分空间（降维操作）
X_train_pca = X_train_std.dot(w)  # 形状从 (124,13) 变为 (124,2)
# 可视化设置
colors = ['r', 'b', 'g']    # 不同类别的颜色
markers = ['s', 'x', 'o']   # 不同类别的标记样式
# 按类别绘制散点图
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],  # 第一主成分值
                X_train_pca[y_train == l, 1],  # 第二主成分值
                c=c, label=l, marker=m)        # 样式参数
plt.xlabel('PC 1 (最大方差方向)')    # X轴标签
plt.ylabel('PC 2 (次大方差方向)')    # Y轴标签
plt.legend(loc='lower left')        # 图例位置
plt.tight_layout()                  # 优化布局
# plt.savefig('images/05_03.png', dpi=300)  # 保存图像
plt.show()  # 显示可视化结果

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    绘制分类器的决策区域可视化图
    参数：
    X : ndarray
        二维特征数据矩阵，形状为(n_samples, 2)
    y : array_like
        目标标签向量，形状为(n_samples,)
    classifier : object
        实现predict方法的分类器实例
    resolution : float, optional
        决策区域网格的分辨率，默认0.02
    返回值：
    None 仅生成matplotlib可视化图像
    """
    # 初始化绘图标记和颜色配置
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # 生成决策面网格数据
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # 预测并绘制决策区域
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # 绘制各分类样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap(idx),
                    edgecolor='black', 
                    marker=markers[idx], 
                    label=cl)
        
        
# 初始化PCA模型，设置降维后保留2个主成分
pca = PCA(n_components=2)
# 对标准化后的训练集进行拟合并转换（学习主成分方向并降维）
X_train_pca = pca.fit_transform(X_train_std)
# 对标准化后的测试集进行转换（使用训练集学习的主成分方向）
X_test_pca = pca.transform(X_test_std)
# 初始化逻辑回归分类器
# multi_class='ovr': 一对多方法处理多分类
# random_state=1: 固定随机种子保证结果可复现
# solver='lbfgs': 使用L-BFGS优化算法（适用于小数据集）
lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
# 使用降维后的训练数据训练分类器
lr = lr.fit(X_train_pca, y_train)
# 绘制分类决策边界可视化图
# X_train_pca: 降维后的训练特征矩阵
# y_train: 训练标签
# classifier=lr: 训练好的逻辑回归模型
plot_decision_regions(X_train_pca, y_train, classifier=lr)
# 设置坐标轴标签  
plt.xlabel('PC 1')  # 第一主成分轴
plt.ylabel('PC 2')  # 第二主成分轴
plt.legend(loc='lower left')  # 图例显示在左下方
plt.tight_layout()   # 自动调整子图间距
# 保存高分辨率图像（需要时取消注释）
# plt.savefig('images/05_04.png', dpi=300)
plt.show()  # 显示可视化结果

# 设置numpy打印精度（保留4位小数）
np.set_printoptions(precision=4)
# ========== 计算各类别均值向量 ==========
mean_vecs = []
for label in range(1, 4):  # 遍历3个葡萄酒类别
    # 计算每个类别的特征均值向量（沿样本轴求平均）
    class_mean = np.mean(X_train_std[y_train == label], axis=0)
    mean_vecs.append(class_mean)
    # 打印各葡萄酒品种的均值向量
    print('MV %s: %s\n' % (label, mean_vecs[label-1]))
# ========== 计算类内散度矩阵S_w ==========
d = 13  # 特征维度（葡萄酒数据集有13个特征）
S_w = np.zeros((d, d))  # 初始化类内散度矩阵
# 遍历每个类别和对应的均值向量
for label, mv in zip(range(1, 4), mean_vecs):
    # 计算当前类别的协方差矩阵（rowvar=False表示列代表特征）
    class_scatter = np.cov(X_train_std[y_train == label], rowvar=False)
    S_w += class_scatter  # 累加各类别的协方差矩阵
# ========== 计算类间散度矩阵S_b ==========
mean_overall = np.mean(X_train_std, axis=0)  # 全局均值向量
S_b = np.zeros((d, d))  # 初始化类间散度矩阵
# 遍历每个类别的均值向量
for i, mv in enumerate(mean_vecs):
    n = X_train_std[y_train == i+1, :].shape[0]  # 当前类别的样本数
    mv = mv.reshape(d, 1)  # 将均值向量转为列向量
    mean_overall = mean_overall.reshape(d, 1)  # 全局均值转为列向量
    # 计算类间散度贡献：(μ_i - μ)(μ_i - μ)^T * n_i
    S_b += n * (mv - mean_overall).dot((mv - mean_overall).T)
# ========== 求解广义特征值问题 ==========
# 计算矩阵S_w^{-1}S_b的特征值和特征向量
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
# 将特征值与对应特征向量组成元组列表
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) 
              for i in range(len(eigen_vals))]
# 按特征值大小降序排序
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# ========== 绘制判别能力可视化 ==========
tot = sum(eigen_vals.real)  # 总判别能力
discr = [i/tot for i in sorted(eigen_vals.real, reverse=True)]  # 各维度判别能力比率
cum_discr = np.cumsum(discr)  # 累积判别能力
plt.bar(range(1,14), discr, alpha=0.5, align='center',
        label='Individual "discriminability"')
plt.step(range(1,14), cum_discr, where='mid',
         label='Cumulative "discriminability"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])  # 设置Y轴范围
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# ========== 构建投影矩阵 ==========
# 取前两个最大判别方向的特征向量（取实数部分）
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
# ========== 数据投影与可视化 ==========
X_train_lda = X_train_std.dot(w)  # 将数据投影到LDA空间
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
# 绘制不同类别的样本点（第二个LD方向取反，仅用于可视化方向调整）
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train == l, 0],
                X_train_lda[y_train == l, 1] * (-1),  # 反转第二个LD方向
                color=c,  # 颜色参数（修正参数名colorizer->color）
                label=l, 
                marker=m)
plt.xlabel('LD 1')  # 第一线性判别方向
plt.ylabel('LD 2')  # 第二线性判别方向
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# 初始化LDA模型，设置降维至2个线性判别成分
lda = LDA(n_components=2)
# 使用标准化训练数据执行LDA转换（监督式降维，需要标签信息）
# 返回降维后的二维特征矩阵 (n_samples, 2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
# 初始化逻辑回归分类器
# multi_class='ovr': 使用一对多策略处理多分类问题
# solver='lbfgs': 适用于小数据集的优化算法
# random_state=1: 固定随机种子保证结果可复现
lr = LogisticRegression(multi_class="ovr", random_state=1, solver='lbfgs')
# 使用降维后的训练数据训练分类器
lr.fit(X_train_lda, y_train)
# 调用自定义函数绘制决策边界可视化图
# X_train_lda: 降维后的二维特征数据
# y_train: 训练标签
# classifier=lr: 训练好的逻辑回归模型
plot_decision_regions(X_train_lda, y_train, classifier=lr)
# 设置坐标轴标签
plt.xlabel('LD 1')  # 第一线性判别方向
plt.ylabel('LD 2')  # 第二线性判别方向
# 在左下方显示图例
plt.legend(loc='lower left')
# 自动调整子图间距避免标签重叠
plt.tight_layout()
# 显示可视化图形
plt.show()


def rbf_kernel_pca(X, gamma, n_components):
    """
    实现基于RBF核的主成分分析（Kernel PCA）
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]
        输入特征矩阵，每行代表一个样本，每列代表一个特征 
    gamma: float
        RBF核的调节参数，控制高斯核的宽度（gamma越大，核越窄） 
    n_components: int
        需要返回的主成分数量
    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, n_components]
        投影后的降维数据集（保留前k个核主成分）

    """
    # 计算样本间平方欧氏距离矩阵（上三角展开形式）
    # pdist返回压缩存储的距离数组（节省空间）
    sq_dists = pdist(X, 'sqeuclidean')
    # 将压缩距离向量转换为对称方阵
    # 得到形状为(n_samples, n_samples)的全距离矩阵
    mat_sq_dists = squareform(sq_dists)
    # 计算RBF核矩阵（高斯核）
    # K_ij = exp(-gamma * ||x_i - x_j||^2)
    K = exp(-gamma * mat_sq_dists)
    # 核矩阵中心化处理（使特征空间数据均值为0）
    N = K.shape[0]  # 样本数量
    one_n = np.ones((N, N)) / N  # 全1矩阵/N
    # 中心化公式：K_centered = K - 1_N*K - K*1_N + 1_N*K*1_N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # 对中心化核矩阵进行特征分解
    # eigh返回升序排列的特征值（需逆序处理）
    eigvals, eigvecs = eigh(K)
    # 将特征值和特征向量按降序排列
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    # 选取前k个特征向量（每个列向量是一个主成分方向）
    # 注意：核PCA的特征向量已对应投影后的样本表示
    X_pc = np.column_stack([eigvecs[:, i] 
                          for i in range(n_components)])
    return X_pc



X,y=make_moons(n_samples=100, random_state=123)
scki_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scki_kpca.fit_transform(X)
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
            color='blue', marker='o', alpha=0.5)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.tight_layout()  
plt.show()





