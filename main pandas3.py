# 导入必要的库
import numpy as np
import pandas as pd

# 创建DataFrame，包含三列数据：
# - a列：1到10的numpy数组
# - b列：1到10的Pandas Series
# - c列：1到10的Python列表
A1 = pd.DataFrame({
    "a": np.arange(1,11),
    "b": pd.Series([1,2,3,4,5,6,7,8,9,10]),
    "c": [1,2,3,4,5,6,7,8,9,10]
})

# 设置行索引为字母A-J
A1.index = ["A","B","C","D","E","F","G","H","I","J"]
print("原始数据：")
print(A1)

# 对a和b列所有值增加10（原地修改）
A1.loc[:, ["a","b"]] += 10
print("\n修改后的数据：")
print(A1)

# 条件筛选（注意运算符优先级问题，建议加括号）
# 筛选条件：a列>15 且 c列<9
A2 = A1[(A1.loc[:, "a"] > 15) & (A1.loc[:, "c"] < 9)]
print("\n筛选结果A2：")
print(A2)

# 输出a列是否>=15的布尔序列
print("\na列是否>=15：")
print(A1.loc[:, "a"] >= 15)

# 使用query方法筛选（等价于A2的条件）
# 筛选条件：a>=12 且 c<21
A3 = A1.query("a >= 12 & c < 21")
print("\n筛选结果A3：")
print(A3)

# 筛选c列值在[1,2,3]中的行
A4 = A1[A1.loc[:, "c"].isin([1,2,3])]
print("\n筛选结果A4（c列包含1-3）：")
print(A4)

# 检查整个DataFrame是否包含1/2/3，返回每行是否包含的布尔值
print("\n任意列包含1/2/3的行：")
print(A1.isin([1,2,3]).any(axis=1))

# 生成数值列的统计描述（默认axis=0，按列统计）
print("\n统计描述：")
print(A1.describe())

# 每行的最小值（axis=1按行计算）
print("\n每行最小值：")
print(A1.min(axis="columns"))

# 每列的最大值（axis=0按列计算）
print("\n每列最大值：")
print(A1.max(axis=0))

# 每行最大值所在的列名
print("\n每行最大值所在列：")
print(A1.idxmax(axis=1))

# 每列最小值所在的行名
print("\n每列最小值所在行：")
print(A1.idxmin(axis=0))

# 按行方向计算累积和（从左到右）
print("\n行累积和：")
print(A1.cumsum(axis=1))

# 按列方向计算累积乘积（从上到下）
print("\n列累积乘积：")
print(A1.iloc[0:6,:].cumprod(axis=0))