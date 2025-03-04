# 导入numpy和pandas库
import numpy as np
import pandas as pd

# 创建Series对象（带自定义索引）
A = pd.Series([1,2,3,4,5,6,7,8,9,10], 
              index=list('ABCDEFGHIJ'))  # 索引使用字母A-J
print("原始Series A:")
print(A)

# 按索引排序（升序，不修改原对象）
A1 = A.sort_index(inplace=False, ascending=True)
# 按值排序（默认升序，不修改原对象）
A2 = A.sort_values(inplace=False)
print("\n按索引排序结果A1:")
print(A1)
print("\n按值排序结果A2:")
print(A2)

# 创建5x4的随机数DataFrame
B = pd.DataFrame(np.random.randn(5,4),  # 标准正态分布随机数
                columns=list('ABCD'),   # 列名为ABCD
                index=list('abcde'))    # 行索引为abcde
print("\nDataFrame B:")
print(B)

# 数据选择演示
print("\niloc选择前三行前两列:")       # 位置索引（左闭右开）
print(B.iloc[0:3, 0:2])              # 0:3 → 行0,1,2；0:2 → 列0,1

print("\nloc标签选择a-c行A-C列:")      # 标签索引（包含两端）
print(B.loc['a':'c', 'A':'C'])       # 包含c行和C列

print("\n条件筛选（A列值>1的行）:")
print(B.loc[B['A'] > 1, :])          # 所有满足条件的行，保留全部列

print("\n混合索引方式1（列位置计算）:") 
print(B.iloc[1:3,                    # 行位置1-2
            B.columns.get_loc('C') : B.columns.get_loc('D')+1])  # 列C到D

print("\n混合索引方式2（获取列位置数组）:")
print(B.iloc[1:3, 
            B.columns.get_indexer(['C','D'])])  # 精确获取C/D列位置

print("\n混合索引方式3（行列都通过索引器获取）:")
print(B.loc[B.index[0:3],            # 前三个索引标签
           B.columns[B.columns.get_indexer(['C','D'])]])  # C/D列

# 数据修改示例
C = pd.DataFrame(np.random.randn(5,4),
                columns=list('ABCD'),
                index=list('abcde'))
print("\n修改前的DataFrame C:")
print(C)

# 通过loc修改数据（标签范围赋值）
C.loc['a':'c', 'A':'C'] = 3  # 修改a-c行，A-C列为固定值3
print("\n修改后的DataFrame C:")
print(C)