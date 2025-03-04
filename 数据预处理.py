# coding: utf-8
import pandas as pd
from io import StringIO
import sys
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 创建包含缺失值的示例CSV数据
# 注意：注释使用单独行，便于read_csv过滤
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
# C列缺失（第二行数据）
5.0,6.0,,8.0  
# D列缺失（第三行数据）
10.0,11.0,12.0,'''
# 使用pandas读取CSV数据，通过comment参数过滤注释行
# 注意：StringIO用于将字符串转换为文件流对象
df = pd.read_csv(StringIO(csv_data), comment='#')
print("原始数据：")
print(df)
# 数据基本信息分析 --------------------------------------------------
# 1. 统计各列缺失值数量（isnull()+sum()组合方法）
# 2. 获取底层numpy数组（df.values属性）
# 3. 删除包含缺失值的行（axis=0表示行方向）
# 4. 删除包含缺失值的列（axis=1表示列方向）
print("\n基础处理：")
print(df.isnull().sum(), df.values, df.dropna(axis=0), df.dropna(axis=1), 
     sep='\n', end='\n\n')
# 高级缺失值处理 --------------------------------------------------
# 1. how='all'：仅删除全为缺失值的行（本示例数据无符合条件行）
# 2. thresh=4：保留至少有4个非缺失值的行（总列数4，即不允许任何缺失）
# 3. subset=['C']：仅在C列存在缺失值时删除对应行
print("\n高级处理：")
print(df.dropna(axis="index", how='all'),      # 等价axis=0
      df.dropna(axis="index", thresh=4),      # 本示例效果同df.dropna()
      df.dropna(axis="index", subset=['C']),  # 删除C列有缺失的第二行
      sep='\n', end='\n\n')
# 使用SimpleImputer进行缺失值填充 ---------------------------------
# missing_values：指定缺失值类型（需与数据中缺失标识一致）
# strategy='mean'：使用列均值填充（也可选median/most_frequent/constant）
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr.fit(df.values)  # 在训练数据上计算各列均值
imputed_data = imr.transform(df.values)  # 应用均值填充
print("\nSimpleImputer填充结果：")
print(imputed_data)
# 使用pandas内置方法填充缺失值 ------------------------------------
# df.fillna()可直接用统计值填充，mean()计算列均值
# 注意：默认返回新对象（inplace=False），若需修改原df需设inplace=True
print("\nDataFrame.fillna填充结果：")
print(df.fillna(df.mean()))  # 此处仅为演示，未保存结果到变量


# ==================== 数据准备阶段 ====================
# 创建示例数据集（三行四列）
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],  # 绿色M号商品，价格10.1，类别2
    ['red', 'L', 13.5, 'class1'],    # 红色L号商品，价格13.5，类别1
    ['blue', 'XL', 15.3, 'class2']   # 蓝色XL号商品，价格15.3，类别2
])
# 设置数据框列名（特征说明）
df.columns = ['color', 'size', 'price', 'classlabel']  # 颜色 | 尺寸 | 价格 | 分类标签
# ==================== 特征工程：尺寸编码 ====================
# 创建有序尺寸映射字典（XL > L > M）
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
# 应用尺寸映射转换（将字符串尺寸转换为数值）
df['size'] = df['size'].map(size_mapping)  # M→1, L→2, XL→3
print("\n尺寸编码后的数据：")
print(df)
# ==================== 尺寸解码演示 ====================
# 创建逆向尺寸字典（数值→字符串）
inv_size_mapping = {v: k for k, v in size_mapping.items()}
# 逆向转换演示（实际应用时需赋值给列才会生效）
df['size'].map(inv_size_mapping)  # 1→M, 2→L, 3→XL
# ==================== 特征工程：类别标签编码 ====================
# 自动生成类别映射字典（基于唯一值排序）
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
# 应用类别编码（class2→0, class1→1，因np.unique排序后class1在前）
df['classlabel'] = df['classlabel'].map(class_mapping)
# ==================== 类别标签解码演示 ====================
# 创建逆向类别字典
inv_class_mapping = {v: k for k, v in class_mapping.items()}
# 逆向转换演示（需赋值才会生效）
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
# ==================== 使用scikit-learn进行标签编码 ====================
# 初始化标签编码器
class_le = LabelEncoder()  # 注意：重复初始化会重置学习到的映射关系
# 自动学习类别并转换（推荐方式，保持编码器状态）
y = class_le.fit_transform(df['classlabel'].values)  # 返回numpy数组



# 提取特征矩阵（包含颜色、尺寸、价格三个特征）
X = df[['color', 'size', 'price']].values
# 创建独热编码器（用于处理分类特征）
color_ohe = OneHotEncoder()
# 对颜色列（索引0）进行独热编码，reshape确保二维输入格式
# 结果转换为稠密数组形式（默认返回稀疏矩阵）
print(color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray())
# 使用ColumnTransformer组合特征处理流程
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),  # 对第0列（颜色）做独热编码
    ('nothing', 'passthrough', [1, 2]) # 保留第1,2列（尺寸、价格）不做处理
])
# 应用转换并转换为float类型（独热编码结果是浮点数）
c_transf.fit_transform(X).astype(float)
# 使用pandas自动进行独热编码（自动识别字符型列）
pd.get_dummies(df[['price', 'color', 'size']])
# 删除第一个哑变量列（避免共线性问题，适用于线性模型）
pd.get_dummies(df[['price', 'color', 'size']], drop_first=True)
# 初始化独热编码器，设置参数：
# - categories='auto'：自动检测特征的所有可能类别
# - drop='first'：删除每个特征的第一列，避免多重共线性问题（适用于线性模型）
color_ohe = OneHotEncoder(categories='auto', drop='first')
# 创建特征列转换器，组合多个预处理步骤：
# 参数是转换器元组列表，每个元组包含：
# - 名称字符串（'onehot'）
# - 转换器实例（color_ohe）
# - 应用转换的列索引（[0] 表示第一列）
c_transf = ColumnTransformer([
    ('onehot', color_ohe, [0]),    # 对索引0列（颜色）进行独热编码
    ('nothing', 'passthrough', [1, 2])  # 保留索引1（尺寸）、2（价格）列不做处理
])
# 应用转换并转换结果为float类型：
# - fit_transform：先学习转换规则，再应用转换
# - astype(float)：将结果转换为浮点型（独热编码结果是0/1的浮点数）
c_transf.fit_transform(X).astype(float)


df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
print('Class labels', np.unique(df_wine['Class label']))


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
