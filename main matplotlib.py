"""
多子图绘制示例 - 对比函数图像
功能:在同一画布中并排展示x²和x^1.5函数图像
"""

# 导入核心库（推荐分两行写更清晰）
import matplotlib.pyplot as plt  # 绘图库，约定缩写为plt
import numpy as np  # 数值计算库，约定缩写为np

# 中文显示配置（必须设置在绘图操作之前）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------- 数据准备 ----------
x = np.arange(0, 10)  # 生成0-9的整数数组，包含10个元素
print("x数组内容:", x)  # 调试输出验证数据

# 计算函数值（注意numpy数组的广播特性）
y = x**2   # 二次函数 
y2 = x**1.5  # 1.5次幂函数

# ---------- 画布初始化 ----------
# 创建1行2列的子图布局（返回画布对象和axes数组）
# figsize: 画布物理尺寸(宽8英寸,高4英寸) 
# dpi: 每英寸像素数(50*8=400像素宽度)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), dpi=50)

# ---------- 公共配置 ----------
# 生成时间标签列表（示例数据）
z = [f"11点{i}分" for i in range(60)]  # 列表推导式生成60个时间字符串

# ---------- 左子图配置 ----------
# 绘制第一条曲线（完整参数演示）
ax1.plot(x, y, 
        color='blue',        # 曲线主色
        linewidth=2,         # 线宽(单位：磅)
        linestyle='--',      # 线型：虚线
        marker='o',          # 数据点标记形状
        markersize=10,       # 标记大小
        markerfacecolor='blue',    # 标记填充色
        markeredgecolor='green',   # 标记边框色
        markeredgewidth=2,   # 边框粗细
        label='y=x^2'        # 图例标签
        )

# 坐标轴设置
ax1.set_xticks(x)  # 设置x轴刻度位置
ax1.set_xticklabels(z[0:60:6])  # 设置x轴标签：从z中每隔6个取一个标签（共10个）
ax1.set_yticks(range(0, 101, 10))  # y轴刻度0-100，步长10
ax1.legend(loc='best')  # 自动选择最佳图例位置

# ---------- 右子图配置 ----------
ax2.plot(x, y2, 
        color='red',         # 使用不同颜色区分曲线
        linewidth=2,
        linestyle='--',
        marker='o',
        markersize=10,
        markerfacecolor='blue',   # 保持相同标记风格
        markeredgecolor='green',
        markeredgewidth=2,
        label='y=x^1.5'     # 不同图例标签
        )

# 坐标轴设置（与左子图对称）
ax2.set_xticks(x)
ax2.set_xticklabels(z[0:60:6])
ax2.set_yticks(range(0, 101, 10))
ax2.legend(loc='best')

# ---------- 全局优化 ----------
plt.tight_layout()  # 自动调整子图间距，防止标签重叠
plt.show()  # 显示图像（注意原代码有重复show()调用）


