"""
NumPy 核心功能全解示例
涵盖数组创建、操作、数学运算、统计计算等关键功能
"""

# ==== 第一部分：基础数组操作 ====
import numpy as np

# 创建3x3整数数组
A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.int64)
print(A.shape,    # 输出形状 (3,3)
      A.ndim,     # 维度数 2
      A.size,     # 元素总数 9
      A.dtype,    # 数据类型 int64
      A.itemsize) # 每个元素字节数 8

# 创建全零数组（显式指定类型）
B = np.zeros((3,3), dtype=np.int64)  # 3x3全0矩阵

# 创建与B形状相同的全1数组（改变数据类型）
C = np.ones_like(B, dtype=np.float64)  # 结果矩阵元素为1.0

# 数组复制方式对比
A1 = np.array(A)     # 深拷贝（完全独立的新数组）
A2 = np.asarray(A)   # 若A是ndarray则视图，否则拷贝
A3 = np.asanyarray(A) # 保持子类类型（如果A是子类）

# ==== 第二部分：序列生成 ====
# 线性等分序列（包含终点）
D1 = np.linspace(0,10,5,endpoint=True)   # [ 0.   2.5  5.   7.5 10. ]

# 线性等分序列（不包含终点）
D2 = np.linspace(0,10,5,endpoint=False)  # [0. 2. 4. 6. 8.]

# 对数等分序列（基数为10）
E1 = np.logspace(0,10,5,endpoint=True)   # [1.e+00 1.e+02 1.e+05 1.e+07 1.e+10]

# 自定义步长序列
F = np.arange(0,10,2,dtype=np.int64)  # [0 2 4 6 8]

# ==== 第三部分：随机数组操作 ====
normal_array = np.random.normal(0, 1, 10)        # 标准正态分布
uniform_array = np.random.uniform(0, 10, 10)     # 均匀分布
int_array = np.random.randint(0, 10, [2,8])         # 随机整数
rand_matrix = np.random.rand(3, 3)               # 3x3矩阵
diag_matrix = np.eye(5, 3, k=1)                  # 5x3对角矩阵,对角线上移1

# 改变数组形状（返回新数组）
I = normal_array.reshape(2,5)  # 必须总元素数匹配原数组

# 原地修改数组形状（可能改变元素总数）
uniform_array.resize(2,5)  # 若新形状元素更少会截断，更多会补0

# ==== 第四部分：索引与切片 ====
"""
假设I数组内容:
[[-0.5  0.3  1.2 -1.1  0.8]
 [ 0.7 -0.9  0.4  1.5 -0.3]]
"""
J1 = I[0,1]    # 取第0行第1列 → 0.3
J2 = I[0,1:3]  # 第0行，1-2列 → [0.3, 1.2]
J3 = I[0:2,1:3] # 1-2行，1-2列 → [[0.3,1.2],[ -0.9,0.4]]
J4 = I[0,0:5:1] # 第0行所有列 → 完整第0行
J5 = I.T        # 转置 → 5x2矩阵

# ==== 第五部分：类型转换与序列化 ====
K1 = I.astype(np.int64)  # 强制类型转换（会丢失小数部分）
K2 = I.tobytes()         # 序列化为字节对象（用于网络传输/存储）

# ==== 第六部分：数组操作 ====
L = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])  # 10x1列向量
L1 = np.array(list(L.T)*10)  # 将行向量重复10次 → 10x10矩阵（每行1-10）
L11 = L1.copy()  # 创建拷贝(L1不会影响L11)

# 条件赋值（原地修改）
L1[L1 >= 5] = 1  # 所有≥5的元素设为1

# 条件索引（返回新数组）
L2 = L11[L11 <= 2]  # 展平后选取≤2的元素 → 1D数组

# 去重排序
L3 = np.unique(L11)  # 返回排序后的唯一值 → [1,2,3,4,5,6,7,8,9,10]

# 逻辑判断
print(np.all(L11 > 3),   # 是否所有元素>3 → False
      np.any(L11 > 3))   # 是否存在元素>3 → True

# ==== 第七部分：数学运算 ====
# 条件运算
cond1 = np.where(np.logical_and(L1>1, L11<3), 1, 0)  # 1<值<3的位置置1
cond2 = np.where(np.logical_or(L11>3, L11<1), 1, 0)  # >3或<1的位置置1

# 统计计算
print(np.max(L11),    # 最大值 10
      np.min(L11),    # 最小值 1
      np.mean(L11),   # 平均值 5.5
      np.median(L11), # 中位数 5.5
      np.std(L11),    # 标准差 2.872
      np.var(L11),    # 方差 8.25
      np.sum(L11),    # 总和 550（10x10矩阵每个位置1-10）
      np.prod(L11))   # 累积乘积（数值极大可能溢出）

# 累积计算
cumsum = np.cumsum(L11)  # 累加序列
cumprod = np.cumprod(L11) # 累乘序列（很快会数值溢出）

# 排序索引
np.argmax(L11, axis=0)  # 每列最大值的索引
np.argmin(L11, axis=1)  # 每行最小值的索引
np.sort(L11, axis=0)    # 按列排序
np.argsort(L11, axis=1) # 每行的排序索引

# ==== 第八部分：矩阵运算 ====
M1 = L11 + 1  # 广播运算：所有元素+1
M2 = L11 * 2  # 所有元素×2
M3 = M1 + M2  # 矩阵逐元素相加

# 不同乘法对比
M4 = np.dot(M1,M2)    # 矩阵乘法（行×列）
M5 = np.multiply(M1,M2) # 逐元素乘法
M6 = np.divide(M1,M2)  # 逐元素除法
M7 = np.mod(M1,M2)     # 逐元素取模     
# 运算类型	外积（np.outer）	                    矩阵乘法（np.dot）
# 数学定义	两个向量的逐元素乘积	                行与列的点积求和
# 维度要求	输入可以是任意形状（自动展平为一维）	第一个矩阵的列数 = 第二个矩阵的行数
# 输出形状	(m, n) （m为arr1元素总数，n为arr2元素总数）	(m, p) （若输入是(m,k)和(k,p)）
M8 = np.power(M1,M2)   # 幂运算
M9 = np.matmul(M1,M2)  # 矩阵乘法（同dot）
M10 = np.outer(M1,M2.T) # 外积运算
