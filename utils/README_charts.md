# Shape折线图生成工具

## 功能概述

该工具可以根据CSV文件中的Shape字段生成详细的折线图，用于可视化不同Shape类型的测量值分布模式。

## 主要功能

1. **每个Shape单独图表** - 为每个Shape类型生成独立的图表文件
2. **显示所有数据线** - 展示原始数据而非聚合均值，保留完整的数据细节
3. **三面板对比** - 每个图表包含三个子图：
   - 左侧：参考文件该Shape的所有数据线
   - 中间：生成文件该Shape的所有数据线
   - 右侧：参考数据和生成数据的叠加对比图
4. **Y轴统一** - 同一个Shape的三个子图使用相同的Y轴范围，便于准确比较
5. **文件覆盖** - 每次生成图表会覆盖之前的同名文件
6. **批量处理** - 自动处理BY、BZ、DY、DZ四个方向
7. **自动适配** - 根据不同方向自动识别对应的数据列

## 使用方法

### 1. 单独生成图表

```python
from shape_line_chart import generate_all_shape_charts

# 生成所有方向的图表
results = generate_all_shape_charts()
```

### 2. 集成到quick_compare

```bash
# 运行比较分析 + 生成图表
python utils/quick_compare.py

# 只运行比较分析，不生成图表
python utils/quick_compare.py --no-charts
```

### 3. 生成特定方向图表

```python
from shape_line_chart import generate_shape_line_chart

# 生成BY方向图表
success = generate_shape_line_chart(
    direction="BY",
    ref_file_path="Data/9600/X9600BY/output.csv",
    gen_file_path="Output/X9600_BY_result.csv",
    output_dir="charts",
    show_plot=False  # 是否显示图表
)
```

## 输出文件

图表保存在 `charts/` 目录下，为每个Shape类型生成独立图表：

### 按方向组织的文件：
- **BY方向**：`X9600_BY_Shape_NN_lines.png`, `X9600_BY_Shape_PN_lines.png`, `X9600_BY_Shape_PP_lines.png`
- **BZ方向**：`X9600_BZ_Shape_NNN_lines.png`, `X9600_BZ_Shape_NNP_lines.png`, 等
- **DY方向**：`X9600_DY_Shape_NNN_lines.png`, `X9600_DY_Shape_NNP_lines.png`, 等
- **DZ方向**：`X9600_DZ_Shape_NNNN_lines.png`, `X9600_DZ_Shape_NNNP_lines.png`, 等

## 图表结构

每个Shape图表包含三个并排的子图：

1. **左侧子图** - 参考文件中该Shape的所有数据线：
   - 蓝色半透明线条，每条线代表一个样本
   - 显示图例（前5条数据线）
   - 包含网格和坐标轴标签

2. **中间子图** - 生成文件中该Shape的所有数据线：
   - 红色半透明线条，每条线代表一个样本
   - 显示图例（前5条数据线）
   - 包含网格和坐标轴标签

3. **右侧子图** - 叠加对比图：
   - 参考数据和生成数据叠加在同一图表中
   - 参考数据：蓝色半透明线条（alpha=0.3）
   - 生成数据：红色半透明线条（alpha=0.3）
   - 便于直观比较两组数据的差异
   - 显示组合图例，标明两组数据的线条数量

## 数据列映射

工具会自动根据方向识别对应的数据列：

- **BY**: `ADD13-D1` 到 `ADD13-D9` (9个点)
- **BZ**: `FAI68-P1` 到 `FAI68-P18` (18个点)
- **DY**: `ADD41-Q1` 到 `ADD41-Q9` (9个点)
- **DZ**: `FAI156-P1` 到 `FAI156-P20` (20个点)

## 颜色说明

- **参考文件数据线**: 蓝色（#1f77b4）半透明，线宽1
- **生成文件数据线**: 红色（#ff7f0e）半透明，线宽1
- **叠加对比图**: 更透明的蓝色和红色线条（alpha=0.3）
- **图例限制**: 为避免图例过于拥挤，前两个子图只显示前5条数据线
- **网格线**: 灰色半透明，提高可读性

## 技术特性

- **Y轴统一**: 三个子图使用相同的Y轴范围，确保视觉比较的一致性
- **文件覆盖**: 每次运行会覆盖同名图表文件，避免文件重复
- **智能布局**: 自动调整子图间距，确保清晰显示
- **鲁棒性**: 处理缺失数据情况，优雅显示"无数据"提示

## 依赖要求

```bash
pip install matplotlib pandas numpy
```

## 注意事项

1. 确保输入文件包含Shape列和对应的数据列
2. 图表使用UTF-8编码支持中文显示
3. 如果数据列不存在，工具会自动跳过该方向
4. 差异图只显示两个文件中都存在的Shape类型

## 示例输出

```
生成所有方向的Shape折线图
==================================================

生成 BY 方向Shape折线图...
  [INFO] 使用 9 个数据列: ['ADD13-D1', 'ADD13-D2', ...]
  [INFO] 找到 3 种Shape类型: ['NN', 'PN', 'PP']
  [SUCCESS] 图表已保存: charts/X9600_BY_shape_lines.png
  BY: 成功

图表生成完成：4/4 成功
  BY: 成功
  BZ: 成功
  DY: 成功
  DZ: 成功
```