# 导入pandas库并使用pd作为别名
import pandas as pd
# 导入numpy库并使用np作为别名
import numpy as np

# 创建Series对象方式1：通过列表创建
# 使用自定义索引（'A'-'J'）
A1 = pd.Series([1,2,3,4,5,6,7,8,9,10], index=list('ABCDEFGHIJ'))

# 创建Series对象方式2：通过numpy数组创建
# 使用默认索引（0-9）
A2 = pd.Series(np.arange(1,11))

# 创建Series对象方式3：通过字典创建
# 字典的键自动成为索引
A3 = pd.Series({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10})

# 打印Series相关信息
print(A1.index,  # 输出索引对象
     A1.values,  # 输出值数组
     A1[1],      # 获取索引位置为1的值（注意不是自定义索引）
     A1,         # 打印整个Series
     sep="\n")   # 用换行分隔输出内容

# 创建DataFrame方式1：通过二维列表创建
B1 = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],
                 columns=['A','B','C'],  # 指定列名
                 index=['a','b','c'])    # 指定行索引

# 创建DataFrame方式2：通过numpy随机数组创建
B2 = pd.DataFrame(np.random.randn(3,3),  # 生成3x3标准正态分布随机数
                 columns=['A','B','C'],
                 index=['a','b','c'])

# 创建DataFrame方式3：通过字典直接创建
# 字典的键成为列名，值自动转换为Series
B3 = pd.DataFrame({'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9]})

# 创建DataFrame方式4：通过显式Series创建
B4 = pd.DataFrame({'A':pd.Series([1,2,3]),
                  'B':pd.Series([4,5,6]),
                  'C':pd.Series([7,8,9])})

# 打印DataFrame相关信息
print(B1.index,    # 行索引
     B1.columns,   # 列名
     B1.shape,     # 形状
     B1.size,      # 元素总数
     B1.ndim,      # 维度数
     B1.dtypes,    # 列数据类型
     B1.axes,      # 行索引和列名
     B1.values,    # 二维数组形式的数据
     B1.iloc[0,0], # 使用位置索引获取第一个元素
     B1.head(2),   # 显示前两行
     B1.tail(2),   # 显示后两行
     sep="\n", end="\n\n")  # 输出分隔和结尾控制

# 打印所有DataFrame对象
print(B1, B2, B3, B4, sep="\n")

# 索引操作示例
C1 = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],
                 columns=['A','B','C'],
                 index=['a','b','c'])

# 修改行索引标签
C1.index = ['d','e','f']


# 重置索引（将原索引转为列，新增默认数字索引）
# drop=False保留原索引列，inplace=True直接修改原对象( 类似于reshape)，inpalce=False会返回修改后的新对象，不改变原对象(类似于resize)
C1.reset_index(drop=False, inplace=True)

# 重置索引并丢弃原索引（但未保存结果）
# 注意：这里没有使用inplace=True，也没有赋值，实际不会生效
C1.reset_index(drop=True)

# 设置'A'列为新索引（drop=True会删除原'A'列）
# 未使用inplace=True且未赋值，不会改变原DataFrame
C1.set_index('A', drop=True)

# 设置'B'列为新索引（保留原'B'列）
# 未保存结果，实际不会生效
C1.set_index('B', drop=False)

# 注意：最后四个操作中只有第一个reset_index真正修改了C1
# 其他操作需要赋值或使用inplace=True才能生效