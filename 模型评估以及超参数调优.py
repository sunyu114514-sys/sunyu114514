import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve, auc
from packaging.version import parse as version
from scipy import __version__ as scipy_version
from numpy import interp
from sklearn.utils import resample
from sklearn.decomposition import KernelPCA as KPCA
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据预处理部分
# ------------------------------------------------------------------
# 载入乳腺癌威斯康星州数据集（WDBC），数据来源：UCI机器学习库
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)
# 特征工程
# 第1-2列为ID和诊断结果，第3列开始是30个细胞核特征
x = df.loc[:, 2:].to_numpy()  # 转换为NumPy数组（569个样本 × 30个特征）
y = df.loc[:, 1].values        # 获取诊断标签列（'M'恶性/'B'良性）
# 标签编码处理
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # 将字符标签转换为数值（B->0, M->1）
print("类别标签映射：", label_encoder.classes_)  # 输出类别对应关系
# 数据集划分
# 使用分层随机划分（保持类别比例），20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.20,
    stratify=y,  # 保持训练集/测试集的类别分布一致
    random_state=1  # 固定随机种子保证可复现性
)
plt.rcParams['axes.unicode_minus'] = False
# 模型构建
# ------------------------------------------------------------------
# 创建机器学习管道（数据预处理 → 特征工程 → 分类器）
pipe_lr = make_pipeline(
    StandardScaler(),          # 数据标准化：使特征均值为0，方差为1
    KPCA(n_components=2,       # 核主成分分析：使用线性核降维到2维
         kernel='linear'),     
    LogisticRegression(        # 逻辑回归分类器
        random_state=1,        # 固定随机种子
        solver='lbfgs',        # 使用L-BFGS优化算法
        max_iter=1000)         # 增加迭代次数确保收敛
)
# 模型训练与评估
# ------------------------------------------------------------------
# 初始模型训练（后续交叉验证会覆盖参数）
pipe_lr.fit(X_train, y_train)
# 测试集性能评估
test_acc = pipe_lr.score(X_test, y_test)
print(f"\n测试集准确率：{test_acc:.3f}")

# 交叉验证流程
# ------------------------------------------------------------------
# 初始化分层10折交叉验证（保持每折的类别分布）
kfold = StratifiedKFold(n_splits=20, shuffle=True, random_state=1)
scores = []
# 执行交叉验证
for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_train, y_train)):
    # 训练阶段
    pipe_lr.fit(X_train[train_idx], y_train[train_idx])
    # 验证阶段
    fold_acc = pipe_lr.score(X_train[test_idx], y_train[test_idx])
    scores.append(fold_acc)
    # 输出当前fold信息
    class_dist = np.bincount(y_train[train_idx])  # 统计训练集的类别分布
    print(f'Fold {fold_idx+1:2d} | 类别分布: {class_dist} | 验证准确率: {fold_acc:.3f}')
# 性能统计
# ------------------------------------------------------------------
print(f"\n交叉验证平均准确率：{np.mean(scores):.3f} ± {np.std(scores):.3f}")
print(f"测试集最终准确率：{test_acc:.3f}")

cv_scores = cross_val_score(pipe_lr, X_train, y_train, 
                           cv=10, n_jobs=-1)
print(f"自动交叉验证结果：{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# 创建包含数据标准化和逻辑回归的机器学习管道
# StandardScaler：标准化处理（均值为0，方差为1）
# LogisticRegression：使用lbfgs优化算法，设置随机种子和最大迭代次数防止不收敛
pipe_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        random_state=1,
        solver='lbfgs',
        max_iter=10000  # 增大迭代次数确保收敛
    )
)
# 生成学习曲线数据
# estimator：使用的评估模型
# train_sizes：定义训练集比例范围（从10%到100%，分10个点）
# cv=10：使用10折交叉验证
# n_jobs=-1：使用所有CPU核心并行计算
train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=10,
    n_jobs=-1
)
# 计算训练/验证集的均值和标准差
# axis=1：沿交叉验证的折数方向计算统计量
train_mean = np.mean(train_scores, axis=1)  # 各训练尺寸的平均训练准确率
train_std = np.std(train_scores, axis=1)    # 标准差反映稳定性
test_mean = np.mean(test_scores, axis=1)    # 各训练尺寸的平均验证准确率
test_std = np.std(test_scores, axis=1)
# 绘制训练准确率曲线
plt.plot(train_sizes, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='Training Accuracy')
# 添加训练准确率置信区间（均值±标准差）
plt.fill_between(train_sizes, 
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')
# 绘制验证准确率曲线（虚线样式）
plt.plot(train_sizes, test_mean,
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='Validation Accuracy')
# 添加验证准确率置信区间
plt.fill_between(train_sizes, 
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
# 图表装饰
plt.grid(linewidth=1, linestyle='-')  # 显示网格线
plt.xlabel('Number of training samples')  # X轴标签
plt.ylabel('Accuracy')  # Y轴标签
plt.legend(loc='lower right')  # 图例位置
plt.ylim([0.8, 1.03])  # 固定Y轴范围便于观察趋势
plt.tight_layout()  # 自动调整子图参数
plt.show()  # 显示图形


# 定义要测试的正则化强度参数C的范围（逆正则化系数，C=1/λ）
# 较小的C值对应更强的L2正则化，防止过拟合
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # 跨越4个数量级

# 生成验证曲线数据
# estimator: 预定义的机器学习管道（包含标准化+逻辑回归）
# param_name: 要调节的管道参数名称（使用双下划线访问管道步骤参数）
# cv=10: 使用10折分层交叉验证
train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,         # 训练集特征
    y=y_train,         # 训练集标签
    param_name='logisticregression__C',  # 管道中LR的C参数
    param_range=param_range,  # 参数测试范围
    cv=10              # 交叉验证折数
)
# 计算训练/验证集的统计量
# axis=1：沿交叉验证的折数方向计算（每个C值对应10折结果）
train_mean = np.mean(train_scores, axis=1)  # 各C值对应的平均训练准确率
train_std = np.std(train_scores, axis=1)    # 标准差反映稳定性
test_mean = np.mean(test_scores, axis=1)    # 各C值的平均验证准确率
test_std = np.std(test_scores, axis=1)
# 绘制训练准确率曲线（蓝色实线）
plt.plot(param_range, train_mean,
         color='blue', marker='o',    # 蓝色圆圈标记
         markersize=5, label='Training Accuracy')
# 添加训练准确率置信带（均值±1标准差）
plt.fill_between(param_range, 
                 train_mean + train_std,
                 train_mean - train_std, 
                 alpha=0.15, color='blue')  # 半透明蓝色区域
# 绘制验证准确率曲线（绿色虚线）
plt.plot(param_range, test_mean,
         color='green', linestyle='--',  # 绿色虚线
         marker='s', markersize=5,       # 方块标记
         label='Validation Accuracy')
# 添加验证准确率置信带
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')
# 图表装饰
plt.grid()  # 显示网格线
plt.xscale('log')  # X轴对数刻度（因C值跨越多个数量级）
plt.legend(loc='lower right')  # 图例显示在右下角
plt.xlabel('Parameter C')       # X轴：正则化强度参数
plt.ylabel('Accuracy')          # Y轴：分类准确率
plt.ylim(0.8, 1.03)             # 固定Y轴范围便于观察
plt.tight_layout()             # 自动调整布局
plt.show()




# 创建支持向量机分类器的处理管道（标准化 → SVM分类器）
# StandardScaler：标准化处理使特征均值为0，方差为1
# SVC：支持向量机分类器，固定随机种子保证可复现性
pip_SVC = make_pipeline(StandardScaler(), SVC(random_state=1))
# 定义SVM参数的搜索范围（跨越6个数量级）
SVM_param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# 配置参数网格（两个核函数的参数组合）
param_grid = [
    # 线性核配置：只需调节正则化参数C
    {"svc__C": SVM_param_range, "svc__kernel": ["linear"]},  # 注意此处应使用小写svc__kernel
    # RBF核配置：需要同时调节C和gamma参数
    {"svc__C": SVM_param_range,
     "svc__gamma": SVM_param_range,  # RBF核的带宽参数
     "svc__kernel": ["rbf"]}
]
# 创建网格搜索对象
gs = GridSearchCV(
    estimator=pip_SVC,    # 要优化的模型管道
    param_grid=param_grid, # 参数搜索空间
    cv=10,                # 10折交叉验证
    scoring="accuracy",   # 使用准确率作为评估指标
    n_jobs=-1,            # 使用所有CPU核心并行计算
    refit=True            # 找到最佳参数后自动用全部数据重新训练
)
# 执行网格搜索（在训练集上进行参数搜索）
gs = gs.fit(X_train, y_train)
# 输出搜索结果
print(f"最佳参数：{gs.best_params_}")   # 显示最优参数组合
print(f"最佳准确率：{gs.best_score_}")  # 显示交叉验证最佳得分
print(f"gs.best_estimator_.score(X_test, y_test):{gs.best_estimator_.score(X_test, y_test)}")#在测试集据上评估最优模型的准确率



# 创建网格搜索对象（内层交叉验证）
# 使用2折交叉验证进行参数选择，减少计算量但可能影响参数稳定性
gs = GridSearchCV(
    estimator=pip_SVC,        # 要优化的SVM管道
    param_grid=param_grid,     # 预定义的参数搜索空间
    cv=2,                     # 内层交叉验证折数（参数选择用）
    scoring="accuracy",       # 优化指标为准确率
    n_jobs=-1,                 # 使用所有CPU核心加速
    refit=True
)
# 执行嵌套交叉验证（外层5折评估）
# 目的：评估「模型+参数搜索」流程的整体性能
scores = cross_val_score(
    estimator=gs,            # 使用包含参数搜索的模型作为评估对象
    X=X_train,               # 使用训练集数据进行嵌套验证
    y=y_train,               # （注意：此处通常应使用完整数据集）
    cv=5,                    # 外层交叉验证折数（性能评估用）
    scoring="accuracy"       # 外层评估指标
)
# 输出嵌套交叉验证结果
print(f"CV准确率：{np.mean(scores):.3f} ± {np.std(scores):.3f}")  # 平均精度及标准差
print("各折详细得分：", scores)  # 显示5个外层折叠的准确率



# 【模型评估与性能可视化】
# 使用网格搜索得到的最佳模型进行预测，并生成混淆矩阵及其可视化
# 执行网格搜索训练（在训练集上寻找最优参数组合）
# 注意：此处gs对象必须已经设置refit=True，才能通过best_estimator_获取最终模型
gs.fit(X_train, y_train)
# 使用最优模型预测测试集
# best_estimator_属性返回的是基于完整训练数据重新训练后的最佳模型
y_pred = gs.best_estimator_.predict(X_test)
# 生成混淆矩阵
# 混淆矩阵格式：
# [[TN, FP]
#  [FN, TP]]
confmat = confusion_matrix(y_test, y_pred)
print("混淆矩阵：\n", confmat)
# 可视化混淆矩阵
fig, ax = plt.subplots(figsize=(2.5, 2.5))  # 创建绘图区域，设置图像大小为2.5x2.5英寸
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)  # 绘制半透明蓝色热力图
# 在每个单元格中添加数值标注
for i in range(confmat.shape[0]):  # 遍历行索引
    for j in range(confmat.shape[1]):  # 遍历列索引
        ax.text(x=j, y=i, 
                s=confmat[i, j],       # 显示矩阵中的具体数值
                va='center',           # 垂直居中对齐
                ha='center')           # 水平居中对齐
# 设置坐标轴标签
plt.xlabel('预测标签\n（0:良性, 1:恶性）')  # X轴：模型预测结果
plt.ylabel('真实标签\n(0:良性, 1:恶性）')  # Y轴：实际标签
# 图表装饰
plt.title('分类结果可视化')  # 添加图表标题
plt.tight_layout()         # 自动调整子图参数，避免内容重叠
plt.show()                 # 显示图形


# 计算并输出测试集上的各项分类指标
# precision: 精确率（预测为恶性的样本中，真实为恶性的比例）
print("precision:%.3f" % precision_score(y_test, y_pred))
# recall: 召回率（真实为恶性的样本中，被正确预测的比例）
print("recall:%.3f" % recall_score(y_test, y_pred))
# f1: F1分数（精确率和召回率的调和平均值）
print("f1:%.3f" % f1_score(y_test, y_pred))



# 定义SVM的正则化参数C的搜索范围（跨越4个数量级）
C_param_range = [ 0.01, 0.1, 1.0, 10.0]
# 配置参数网格（包含两种核函数的参数组合）
param_grid = [
    # 线性核配置：只需调节正则化参数C（控制过拟合）
    {"svc__C": C_param_range, "svc__kernel": ["linear"]},
    # RBF核配置：同时调节C和gamma（控制核函数带宽）
    {"svc__C": C_param_range,
     "svc__gamma": C_param_range,  # gamma值越小，决策边界越平滑
     "svc__kernel": ["rbf"]}
]
# 创建自定义评分器（以类别0[良性]作为正类计算F1分数）
# pos_label=0：因为在医学场景中可能更关注良性肿瘤的准确识别
scorer = make_scorer(f1_score, pos_label=0)
# 初始化网格搜索对象
gs = GridSearchCV(
    estimator=pip_SVC,  # 使用SVM管道
    param_grid=param_grid,  # 参数搜索空间
    cv=10,  # 10折交叉验证
    scoring=scorer,  # 使用自定义F1评分
    n_jobs=-1,  # 使用所有CPU核心并行计算
    refit=True  # 自动用最佳参数重新训练最终模型
)
# 执行网格搜索（在训练集上寻找最优参数组合）
gs.fit(X_train, y_train)
# 输出最佳参数组合和对应的交叉验证得分
print(f"最佳参数：{gs.best_params_}")  # 显示最优C值、gamma值和核类型
print(f"最佳交叉验证F1分数：{gs.best_score_}")  # 基于类别0的F1分数

# 创建一个包含数据预处理和逻辑回归模型的管道
# StandardScaler：标准化处理（均值为0，方差为1）
# PCA(n_components=2)：降维到2个主成分
# LogisticRegression：使用L2正则化，C=100.0（较小的正则化强度）
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(penalty='l2', random_state=1, 
                      solver="lbfgs", C=100.0)
)
# 选择第4和第14列特征作为训练数据（可能是特定领域知识选择的特征）
X_train2 = X_train[:, [4, 14]]
# 创建分层3折交叉验证拆分器（保持类别分布）
# random_state=1 保证可重复性
cv = list(StratifiedKFold(random_state=1, n_splits=3,shuffle=True).split(X_train, y_train))
# 初始化绘图画布
fig = plt.figure(figsize=(7, 5), dpi=500)
# 初始化平均真阳性率（TPR）和固定假阳性率（FPR）范围
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)  # 生成0到1之间等间距的100个点
all_tpr = []
for i, (train_idx, test_idx) in enumerate(cv):  # 添加逗号修正语法
    # 训练模型并预测概率
    probas = pipe_lr.fit(X_train2[train_idx], 
                        y_train[train_idx]).predict_proba(X_train2[test_idx])
    # 计算ROC曲线指标
    fpr, tpr, thresholds = roc_curve(y_train[test_idx],
                                    probas[:, 1],  # 使用类别1（恶性）的概率
                                    pos_label=1)
    # 线性插值使所有ROC曲线具有相同长度的FPR点
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0  # 强制曲线从(0,0)开始 
    # 计算AUC值
    roc_auc = auc(fpr, tpr)
    # 绘制当前fold的ROC曲线
    plt.plot(fpr, tpr, lw=1, 
            label='ROC fold %d (area=%0.2f)' % (i+1, roc_auc))
# 绘制随机猜测的参考线（对角线）
plt.plot([0, 1], [0, 1], linestyle='--', 
        color=(0.6, 0.6, 0.6), label='Random Guessing')
# 计算平均ROC曲线
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0  # 确保曲线结束于(1,1)
mean_auc = auc(mean_fpr, mean_tpr)
# 绘制平均ROC曲线
plt.plot(mean_fpr, mean_tpr, linestyle='-', color='b',
        label='Mean ROC (area=%0.2f)' % mean_auc, lw=2)
# 绘制理想性能参考线（直角线）
plt.plot([0, 0, 1], [0, 1, 1],  # x坐标：[0,0,1]，y坐标：[0,1,1]
        linestyle=':', color='black', lw=2,
        label='Perfect Performance')
# 设置坐标轴范围和标签
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

X_imb = np.vstack((x[y == 0], x[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))
print('Original class distribution:\n{}'.format(np.bincount(y_imb)))
