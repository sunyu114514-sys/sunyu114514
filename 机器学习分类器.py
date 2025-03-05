# 导入机器学习相关库
from sklearn import datasets               # 加载标准数据集
import numpy as np                        # 数值计算库
from sklearn.model_selection import train_test_split  # 数据集分割
from sklearn.preprocessing import StandardScaler      # 数据标准化
from sklearn.linear_model import Perceptron           # 感知机模型
from sklearn.metrics import accuracy_score            # 准确率计算
from matplotlib.colors import ListedColormap          # 颜色映射(后续可视化使用)
import matplotlib.pyplot as plt                       # 可视化绘图
import matplotlib                                     # 基础绘图库
from packaging.version import parse as parse_version  # 版本解析工具(替代已弃用的distutils)
from sklearn.linear_model import LogisticRegression   # 逻辑回归
from sklearn.svm import SVC                           # 支持向量机
from sklearn.linear_model import SGDClassifier        # 随机梯度下降分类
from sklearn.tree import DecisionTreeClassifier       # 决策树分类
from sklearn import tree                              # 树模型工具
from pydotplus import graph_from_dot_data             # 决策树可视化工具
from sklearn.tree import export_graphviz              # 导出决策树图
from sklearn.ensemble import RandomForestClassifier   # 随机森林
from sklearn.neighbors import KNeighborsClassifier    # K近邻算法
from sklearn.multiclass import OneVsRestClassifier

# 中文显示配置(必须设置在绘图操作之前)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 数据准备
iris = datasets.load_iris()  # 🟢加载鸢尾花数据集
X = iris.data[:, [2, 3]]     # 🟢仅使用花瓣长度(第3列)和宽度(第4列)作为特征
y = iris.target              # 🟢目标变量(花的种类)
# 数据集分割(分层抽样)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,           # 🟢测试集占比30%
    random_state=1,          # 🟢随机种子保证可重复性
    stratify=y)              # 🟢分层抽样保持类别比例
# 数据标准化
scaler = StandardScaler()                     # 🟢创建标准化器
X_train_std = scaler.fit_transform(X_train)   # 🟢训练集拟合并转换
X_test_std = scaler.transform(X_test)         # 🟢测试集仅进行转换

# 可视化函数(决策边界绘制)
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """🟢决策区域可视化函数
    参数：
    X : 特征矩阵
    y : 目标向量  
    classifier : 训练好的分类器对象
    test_idx : 测试集索引范围
    resolution : 网格分辨率(值越小图越精细)
    """
    # 初始化标记和颜色
    markers = ('s', 'X', 'o', '^', 'v')  # 🟢不同类别的标记形状
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  # 🟢颜色方案
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 🟢根据类别数创建颜色映射
    # 计算网格边界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 🟢特征1的范围
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 🟢特征2的范围
    # 生成网格点矩阵
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),  # 🟢x轴坐标矩阵
        np.arange(x2_min, x2_max, resolution))  # 🟢y轴坐标矩阵
    # 预测整个网格的类别
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)  # 🟢展平后预测
    Z = Z.reshape(xx1.shape)  # 🟢恢复为网格形状
    # 绘制决策边界
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)  # 🟢填充等高线
    # 绘制样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],        # 🟢按类别着色
                    marker=markers[idx],  # 🟢按类别选择标记
                    label=cl, 
                    edgecolor='black')    # 🟢点边缘颜色
    # 高亮测试集样本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none',            # 🟢无填充颜色
                    edgecolor='black',   # 🟢黑色边框
                    alpha=1.0,           # 🟢完全不透明
                    linewidth=1,         # 🟢边框线宽
                    marker='o',          # 🟢圆形标记
                    s=100,               # 🟢标记尺寸
                    label='test set')    # 🟢图例标签

#============== 感知机分类器 ==============#
"""
原理说明：
1. 单层神经网络结构(输入层+输出层)
2. 激活函数：阶跃函数(输出1或-1)
3. 权重更新规则：w += η(y_i - ŷ_i)x_i
4. 学习率η控制参数更新步长
5. 适用于线性可分数据
"""
# 模型训练
ppn = Perceptron(
    eta0=0.1,          # 🟢学习率η
    random_state=1     # 🟢随机种子
)
ppn.fit(X_train_std, y_train)  # 🟢在标准化数据上训练
# 可视化决策边界
X_combined_std = np.vstack((X_train_std, X_test_std))  # 🟢合并训练集和测试集
y_combined = np.hstack((y_train, y_test))              # 🟢合并目标变量
plot_decision_regions(X_combined_std, y_combined, ppn, test_idx=range(105,150))
plt.title('Perceptron Decision Boundaries')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend()
plt.show()

#============== 逻辑回归 ==============#
"""
原理说明：
1. 使用sigmoid函数将线性组合映射到(0,1)概率区间：σ(z) = 1/(1+e^-z)
2. 损失函数：交叉熵损失 -Σ[y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i)]
3. 正则化参数C=1/λ，防止过拟合
4. 多分类策略：OvR(一对多)
"""
# 模型训练与可视化
# 🟢多分类策略：One-vs-Rest(新版OneVsRestClassifier封装)
lr = OneVsRestClassifier(LogisticRegression(
    C=100.0,             # 🟢正则化强度的倒数(值越大正则化越弱)
    solver='lbfgs',      # 🟢优化算法：有限内存BFGS算法
))
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, lr, test_idx=range(105,150))
plt.title('Logistic Regression Decision Boundaries')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend()
plt.show()
# 正则化效果分析
weights, params = [], []
for c in np.arange(-5, 5):  # 🟢遍历C参数的指数范围
    lr = LogisticRegression(
        C=10.**c,           # 🟢C=10^c 
        solver='lbfgs',
        multi_class='ovr'# 🟢多分类策略：One-vs-Rest(已弃用，建议使用OneVsRestClassifier封装)
)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])  # 🟢存储第二类的权重系数
    params.append(10.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')  # 🟢对数坐标显示
plt.title('Regularization Effect Analysis')
plt.legend()
plt.show()

#============== 支持向量机 ==============#
"""
原理说明：
1. 最大间隔分类器：寻找最优超平面使分类间隔最大化
2. 核技巧：通过核函数将数据映射到高维空间
3. 正则化参数C：权衡间隔大小和分类错误
4. 核函数类型：
   - linear: 线性核
   - rbf: 高斯核(处理非线性可分)
"""
# 线性SVM
svm_linear = SVC(
    kernel='linear',   # 🟢线性核函数
    C=1.0,             # 🟢正则化参数
    random_state=1
)
svm_linear.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, svm_linear, test_idx=range(105,150))
plt.title('Linear SVM Decision Boundaries')
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend()
plt.show()

# RBF核SVM(演示不同gamma值)
X_xor = np.random.randn(200, 2)  # 🟢生成随机数据集
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)  # 🟢创建异或关系
y_xor = np.where(y_xor, 1, -1)  # 🟢转换为1/-1标签
gamma_values = [0.1, 1, 10, 100]  # 🟢不同gamma参数值
for gamma in gamma_values:
    svm_rbf = SVC(
        kernel='rbf',     # 🟢径向基核函数
        gamma=gamma,      # 🟢控制决策边界弯曲程度
        C=10.0            # 🟢正则化参数
    )
    svm_rbf.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, svm_rbf)
    plt.title(f'RBF SVM (gamma={gamma})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))  # 🟢基尼不纯度计算(注意：此实现为简化版，标准公式应为 1 - p² - (1-p)²)
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))  # 🟢信息熵计算(单位：比特)
def error(p):
    return 1 - np.max([p, 1 - p])  # 🟢分类错误率计算(取最大类别概率的补)
# 生成概率值区间 [0,1) 的等差数列，步长0.01
x = np.arange(0.0, 1.0, 0.01)  # 🟢创建0到1之间间隔0.01的数组(共100个点)
# 计算不同指标值
ent = [entropy(p) if p != 0 else None for p in x]  # 🟢处理p=0时的log计算异常
sc_ent = [e * 0.5 if e else None for e in ent]     # 🟢熵值缩放(用于可视化对比)
err = [error(i) for i in x]                        # 🟢计算所有点的分类错误率
# 创建画布和坐标系
fig = plt.figure()
ax = plt.subplot(111)  # 🟢1x1网格的第1个子图(经典单图布局)
# 循环绘制四种指标曲线
for i, lab, ls, c, in zip(
    [ent, sc_ent, gini(x), err],                   # 🟢数据集列表
    ['Entropy', 'Entropy (scaled)',                # 🟢图例标签
     'Gini impurity', 'Misclassification error'],  # 🟢(注意：实际颜色顺序需对应数据)
    ['-', '-', '--', '-.'],                        # 🟢线型：实线、实线、虚线、点划线
    ['black', 'lightgray', 'red', 'green', 'cyan']):# 🟢颜色方案(实际使用前4种)
    line = ax.plot(x, i, label=lab, linestyle=ls, linewidth=2, color=c)  
# 图例配置
ax.legend(loc='upper center', 
         bbox_to_anchor=(0.5, 1.15),  # 🟢将图例定位在画布上方(x轴中心，y轴1.15倍高度)
         ncol=5,                       # 🟢分5列显示(实际4条曲线足够单行显示)
         fancybox=True,                # 🟢圆角边框
         shadow=False)                 # 🟢无阴影
# 添加参考线
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')  # 🟢0.5水平虚线
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')  # 🟢1.0水平虚线
# 坐标轴设置
plt.ylim([0, 1.1])          # 🟢y轴范围(留出0.1空白)
plt.xlabel('p(i=1)')        # 🟢x轴标签：类别1的概率
plt.ylabel('impurity index')# 🟢y轴标签：不纯度指标
#plt.savefig('images/03_19.png', dpi=300, bbox_inches='tight')  # 🟢保存高清图(注释状态)
plt.show()




#============== 决策树分类器 ==============#
"""
原理说明：
1. 树形结构模型：通过递归划分特征空间构建决策规则树
   - 内部节点：特征判断条件(如petal_length ≤ 2.45)
   - 叶节点：最终分类结果
2. 分裂标准：
   - 基尼不纯度(Gini impurity)：衡量节点纯度，计算式为 Gini = 1 - Σ((p|i)²)
   - 信息增益(Information gain)：基于信息熵的减少量(本示例未使用)
3. 停止条件控制：
   - max_depth=4：限制树的最大深度防止过拟合
   - min_samples_split：节点继续分裂的最小样本数(默认2)
4. 特性：
   - 非参数模型：不对数据分布做假设
   - 特征重要性：自动评估特征贡献度
   - 可视化友好：树结构直观可解释
5. 注意事项：
   - 容易产生过拟合(需通过剪枝/参数限制控制)
   - 对噪声数据敏感(需配合数据预处理)
"""
# 创建决策树分类器实例
tree_model = DecisionTreeClassifier(
    criterion='gini',    # 🟢分裂标准：基尼不纯度(衡量节点纯度，值越小纯度越高)
    max_depth=4,         # 🟢树的最大深度(防止过拟合，控制模型复杂度)
    random_state=1       # 🟢随机种子(保证可复现性)
)
# 模型训练(使用原始特征数据，决策树不需要标准化)
tree_model.fit(X_train, y_train)  # 🟢输入训练集特征和标签
# 合并训练集和测试集数据(用于完整可视化)
X_combined = np.vstack((X_train, X_test))  # 🟢垂直堆叠特征矩阵
y_combined = np.hstack((y_train, y_test))  # 🟢水平拼接目标向量
# 绘制决策边界可视化
plot_decision_regions(
    X_combined, y_combined, 
    classifier=tree_model,        # 🟢传入训练好的决策树模型
    test_idx=range(105, 150)      # 🟢高亮显示测试集样本(索引105-149)
)
# 图表设置
plt.xlabel('petal length [cm]')   # 🟢x轴标签(原始单位)
plt.ylabel('petal width [cm]')    # 🟢y轴标签(原始单位)
plt.legend(loc='upper left')      # 🟢图例位置：左上角
plt.tight_layout()                # 🟢自动调整子图参数(避免标签重叠)
# plt.savefig('images/03_20.png', dpi=300)  # 🟢保存高清图像(可选)
plt.show()                        # 🟢显示图表
# 可视化决策树结构
tree.plot_tree(tree_model)        # 🟢绘制树形结构(节点包含分裂条件和基尼值)
# plt.savefig('images/03_21_1.pdf')  # 🟢保存为矢量图(可选，适合印刷)
plt.show()                       # 🟢显示树形图


#============== 随机森林分类器 ==============#
"""
原理说明：
1. 集成学习方法：通过构建多棵决策树进行投票决策
   - 每棵树使用Bootstrap抽样(有放回抽样)训练
   - 特征随机选择：分裂时随机选择部分特征进行考察
2. 核心参数：
   - n_estimators=25：森林中决策树的数量
   - criterion='gini'：节点分裂标准（基尼不纯度）
3. 优势特性：
   - 内置特征重要性评估：通过特征的平均纯度提升计算
   - 袋外数据(OOB)评估：可用oob_score=True开启（本示例未使用）
   - 并行计算：n_jobs参数控制使用的CPU核心数
4. 注意事项：
   - 树的数量越多通常效果越好，但会增加计算成本
   - 随机性来源：数据抽样随机性 + 特征选择随机性
"""
# 创建随机森林分类器实例
forest = RandomForestClassifier(
    criterion='gini',     # 🟢分裂标准：基尼不纯度
    n_estimators=20,      # 🟢森林中树的数量（默认100）
    random_state=1,       # 🟢控制Bootstrap抽样和特征选择的随机性
    n_jobs=12              # 🟢并行使用的CPU核心数（-1表示使用全部）
)
# 模型训练（使用原始特征数据，与决策树相同）
forest.fit(X_train, y_train)  # 🟢输入训练集特征和标签
# 可视化决策边界（使用合并后的原始尺度数据）
plot_decision_regions(
    X_combined, y_combined,
    classifier=forest,        # 🟢传入训练好的随机森林模型
    test_idx=range(105, 150)  # 🟢高亮显示测试集样本(索引105-149)
)
# 图表设置
plt.xlabel('petal length [cm]')  # 🟢x轴标签（原始单位厘米）
plt.ylabel('petal width [cm]')   # 🟢y轴标签（原始单位厘米）
plt.legend(loc='upper left')     # 🟢图例位置：左上角
plt.tight_layout()               # 🟢自动调整子图参数（避免标签重叠）
# plt.savefig('images/03_22.png', dpi=300)  # 🟢保存高清图像（可选）
plt.show()                       # 🟢显示图表



#============== K近邻分类器 ==============#
"""
原理说明：
1. 基于实例的学习：不构建显式模型，通过存储训练数据进行预测
2. 距离度量：使用闵可夫斯基距离（Minkowski distance）
   - p=1：曼哈顿距离（L1范数）
   - p=2：欧氏距离（L2范数，默认）
3. 决策规则：多数投票法（分类）或平均值（回归）
4. 核心参数：
   - n_neighbors=5：考虑最近5个邻居的标签
   - weights='uniform'：默认等权投票（可选'distance'按距离加权）
5. 特性：
   - 无需训练阶段（惰性学习）
   - 对数据规模敏感（需标准化处理）
   - 高维数据效率下降（维度灾难）
"""
# 创建KNN分类器实例 
knn = KNeighborsClassifier(
    n_neighbors=5,   # 🟢近邻数量（奇数可避免平票）
    p=2,             # 🟢距离度量参数（2=欧氏距离）
    metric='minkowski' # 🟢闵可夫斯基距离（p=2时等效欧氏距离）
)
# 模型训练（实际只需存储标准化后的数据）
knn.fit(X_train_std, y_train)  # 🟢输入标准化后的训练数据
# 可视化决策边界（使用标准化后的合并数据）
plot_decision_regions(
    X_combined_std,           # 🟢标准化后的特征矩阵
    y_combined,               # 🟢合并后的目标变量
    classifier=knn,           # 🟢传入训练好的KNN模型
    test_idx=range(105, 150)  # 🟢高亮测试集样本(索引105-149)
)
# 图表设置
plt.xlabel('petal length [standardized]')  # 🟢x轴标签（标准化单位）
plt.ylabel('petal width [standardized]')   # 🟢y轴标签（标准化单位）
plt.legend(loc='upper left')               # 🟢图例位置：左上角
plt.tight_layout()                         # 🟢自动调整布局（防止标签重叠）
# plt.savefig('images/03_24.png', dpi=300)  # 🟢保存高清图像（可选）
plt.show()    