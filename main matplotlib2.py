# 导入必要的库（推荐分开展示更规范）
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库

# 设置中文字体显示（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei是Windows系统自带黑体

# 创建数据
x = np.array([1,2,3,4,5,6,7,8,9,10])  # 创建x轴数据数组
y = x ** 2  # 生成二次函数数据（注意幂运算符号）
y2 = x ** 1.5  # 生成1.5次幂函数数据

# 初始化画布
plt.figure(
    figsize=(4,4),  # 图形尺寸（英寸）：宽4英寸，高4英寸
    dpi=50  # 分辨率：每英寸50像素（总像素200x200）
)

# 创建时间标签（演示动态时间标注）
z = [f"11点{i}分" for i in range(60)]  # 生成60个时间字符串的列表

# 设置x轴刻度（重要参数说明）
plt.xticks(
    ticks=x,  # 刻度位置（使用x数组的值）
    labels=z[0:60:6]  # 标签内容：从z列表每6个取一个，共10个标签
)
#每一个x的值被对应地替换为Z列表的值

# 绘制第一条曲线（完整参数说明）
plt.plot(
    x, y,  # x和y数据
    color='blue',  # 线条颜色
    linewidth=2,  # 线条粗细（单位：磅）
    linestyle='--',  # 线型：虚线
    marker='o',  # 数据点标记形状：圆形
    markersize=10,  # 标记大小
    markerfacecolor='blue',  # 标记填充颜色
    markeredgecolor='green',  # 标记边框颜色
    markeredgewidth=2,  # 标记边框粗细
    label='y=x^2'  # 图例标签
)

# 绘制第二条曲线（参数与第一条对比）
plt.plot(
    x, y2,
    color='red',  # 不同颜色区分曲线
    linewidth=2,
    linestyle='--',
    marker='o',
    markersize=10,
    markerfacecolor='blue',  # 保持相同标记填充色
    markeredgecolor='green',
    markeredgewidth=2,
    label='y=x^1.5'  # 不同图例标签
)

# 添加图例（位置参数说明）
plt.legend(
    loc=0  # 自动选择最佳位置，等价于'best'
    # 其他常用位置参数：
    # 1: 右上角  2: 左上角  3: 左下角  4: 右下角
    # 'upper center', 'lower left' 等字符串形式也可用
)

# 设置y轴刻度范围
plt.yticks(
    range(0, 101, 10)  # 从0到100，每隔10个单位设置一个刻度
)
#不替换成标签，直接采用刻度值

# 显示图形
plt.show()  # 渲染并弹出图形窗口