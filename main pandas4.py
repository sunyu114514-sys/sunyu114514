# 导入numpy和pandas库
import numpy as np  # 虽然导入但未实际使用
import pandas as pd

# 数据读取与基本处理 ============================================
# 读取Excel文件中的指定列（"我"和"是"列）
A = pd.read_excel(
    r"C:\Users\sunyu\Desktop\学习\python\test.xlsx",
    usecols=["我", "是"]  # 仅读取这两列
)
print("原始数据A：")
print(A)

# 数据输出 ====================================================
# 将"我"列数据写入新Excel文件
A.to_excel(
    r"C:\Users\sunyu\Desktop\学习\python\test2.xlsx",
    columns=["我"],          # 只保留"我"列
    header=True,            # 保留列标题
    index=False,            # 不保存行索引
)

# 数据预处理 ==================================================
# 读取原始Excel文件，指定第一列为索引
B = pd.read_excel(
    r"C:\Users\sunyu\Desktop\学习\python\test.xlsx",
    index_col=0  # 将第一列作为行索引
)
print("\n原始数据B：")
print(B)

# 数据清洗 ====================================================
# 尝试删除包含空值的列（但未保存结果，inplace=False）
print("\n尝试删除空值列后的B（未真正修改原始数据）：")
print(B.dropna(axis="columns", how='any', inplace=False))

# 空值检测
print("\n非空值检测结果：")
print(pd.notnull(B))  # 返回布尔矩阵，显示每个位置是否非空

# 数据填充与替换 ==============================================
# 创建填充和替换后的副本（未修改原始数据）
filled_B = B.fillna(value=0, inplace=False)  # 用0填充空值
modified_B = filled_B.replace(to_replace=6, value=144, inplace=False)  # 将6替换为144
print("\n填充和替换后的B（副本）：")
print(modified_B)

# 数据分箱 ====================================================
# 选择"我"列数据
me = B.loc[:, "我"]  # 等效于 me = B["我"]

# 等频分箱（分3个区间）
me1 = pd.qcut(me, 3, labels=["小", "中", "大"])  # 每个区间尽可能包含相同数量的样本
print("\n等频分箱结果：")
print(me1)
print("分箱分布统计：")
print(me1.value_counts())  # 显示每个分箱的样本数量

# 自定义区间分箱
bin_edges = [-1000, 2, 4, 7, 9, 12, 10000]  # 自定义区间边界
binned = pd.cut(me, bin_edges, right=False)  # 左闭右开区间
print("\n自定义分箱结果：")
print(binned)
print("自定义分箱分布统计：")
print(binned.value_counts())  # 显示每个区间的样本数量

# 特征编码 ====================================================
# 尝试进行One-Hot编码（存在问题，参数需调整）
try:
    # 错误示例：columns参数应该接收列名列表
    print(pd.get_dummies(me1, columns="我", prefix="田所浩二"))
except Exception as e:
    print("\n编码错误：", e)

# 正确用法（无需columns参数，直接对Series编码）：
encoded = pd.get_dummies(me1, prefix="等级")
print("\n正确的One-Hot编码结果：")
print(encoded)

# 数据合并 ====================================================
# 将原始数据和分箱结果合并
C = pd.concat([me, me1], axis="columns")  # 按列合并
print("\n合并后的数据集C：")
print(C)