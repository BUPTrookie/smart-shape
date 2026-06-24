"""
测试Matplotlib后端修复
"""

import matplotlib
print(f"默认后端: {matplotlib.get_backend()}")

# 修改为非交互式后端
matplotlib.use('Agg')
print(f"修改后后端: {matplotlib.get_backend()}")

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 测试绘图
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Test Chart')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)

# 保存文件（不显示）
plt.savefig(r'c:\Users\63258\Desktop\综合实践\Code\test_plot.png', dpi=100)
plt.close()

print("[OK] Test passed! Chart saved to test_plot.png")
print("[OK] No Tkinter error occurred")
