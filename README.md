# X9600 铁路产品整形加工数据分析

本项目用于 X9600 系列铁路产品的**来料形状分类**与**整形压头影响预测**，辅助整形工艺决策。核心目标：将产品固定后下压，使其 20 个测量点的最大-最小差 ≤ 0.1。

## 两大核心模块

### 模块A：分BIN算法（来料分类）
把 20 个测量点的曲线压缩为 4 段特征 → Shape 标签 → BIN 分类，用于判断「这根产品该怎么整形」。

| 文件 | 说明 |
|------|------|
| `rail_binning_algorithm.py` | 分BIN核心算法（重构版，`RailBinningCore` 类）：4段特征值 + P/N/M 标签 + BIN 分配 |
| `rail_binning_algorithm_v4.py` | V4 版（`RailBinningCoreV4`）：段3 改为物理4类分类（FLAT/ARC_UP/ARC_DOWN/WAVE） |
| `constants/product_configs.py` | 各方向分段配置（DZ四段 / BY两段 / BZ三段 / DY三段）、阈值、方法 |
| `constants/bin_categories.py` | BIN 规则（整体值→BINOK/BIN100，Shape→BIN1-16，MMM→BIN17/18，共20种） |
| `constants/field_definitions.py` | 字段定义（整体值字段、测量点） |
| `constants/classification_labels.py` | P/N/M 标签定义 |

**算法要点**：段1 特征 = P1−P4（端点差值法）；段2/3 = 直线度拟合（保留符号，取绝对值最大偏差）；段4 = P20−P17。最小二乘拟合值 < 0.018 时前3段记 MMM → BIN17/BIN18。

### 模块B：压头影响分析（整形效果预测）
从历史 Pre/Post 配对数据学习整形压头（RS1–RS4）对 20 个点位的影响规律，训练 Ridge 回归模型预测整形后变化量。

| 文件 | 说明 |
|------|------|
| `rs_impact_analyzer.py` | 压头影响分析主流程（`RSImpactAnalyzer`，5类流水线：DataLoader/FeatureEngineer/ModelTrainer/ResultExporter/编排） |
| `rs_impact_analyzer_v2.py` | RMSE 优化探索版（本地保留、不入库；结论不采纳，生产用 v1） |
| `rs_impact_config.py` | 配置（数据路径、特征工程开关、模型参数、优化策略） |
| `scripts/analyze_rs_segments.py` | 对 Reshaping 表做 RS 分段统计 |

**算法要点**：按 `Barcode` 配对 Pre/Post，Δ=Post−Pre；构造 26 个特征（13位置 + 7 Pre曲线 + 6交互）；`MultiOutputRegressor + Ridge`，5折交叉验证选 alpha，默认 85/15 训练测试划分。

## 基本用法

```bash
# 分BIN分类（以 DZ 方向为例）
python -c "from rail_binning_algorithm import RailBinningCore; import pandas as pd; \
RailBinningCore('X9600_DZ').process(pd.read_csv('Data/X9600DZ/data.csv'))"

# 压头影响分析（运行完整流水线）
python rs_impact_analyzer.py
```

## 在线服务（FastAPI + DB）

把训练好的压头影响模型包装为 HTTP 服务，实时预测整形变化量（参考 smart_shape 的在线服务流程，用 FastAPI + 数据库重写）。

**三步启动：**

```bash
# 1. 训练并持久化模型（生成 model.pkl/scaler.pkl/feature_names.json）
python rs_impact_analyzer.py

# 2. 初始化数据库（建表 + 登记模型版本）
python -m db.init_db

# 3. 启动在线服务（交互式文档见 http://localhost:8000/docs）
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**API：**

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/predict` | 输入压头参数 + Pre 曲线，返回 20 点位 Δ 预测 |
| GET | `/health` | 健康检查 + 当前模型版本 |
| GET | `/history?limit=N` | 最近 N 条预测记录及反馈状态 |
| POST | `/feedback` | 回写实际整形结果，形成闭环 |

**关键文件：** `predictor.py`（模型常驻推理）、`app.py`（FastAPI 入口）、`api/`（路由+Pydantic）、`db/`（SQLAlchemy+SQLite，4张表）。

## 目录结构

```
Code/
├── 核心代码(根级，平级 import)
│   rail_binning_algorithm.py / _v4.py        # 分 BIN（基础版 + V4）
│   rs_impact_analyzer.py / _v2.py            # 压头影响分析（v1 可信 + v2 探索）
│   rs_impact_config.py                       # 配置
│   shaping.py / planner.py / predictor.py    # 案例匹配 / 方案生成 / 推理
│   app.py / paths.py                         # FastAPI 入口 / 统一路径
├── api/          # 路由 + Pydantic 模型
├── db/           # SQLAlchemy + SQLite（4 张表）
├── constants/    # 算法常量（字段/标签/BIN/产品配置）
├── tests/        # 单元测试（pytest，pyproject 设 pythonpath=["."]）
├── scripts/      # 可视化/数据处理/评估脚本（python -m scripts.xxx 运行）
├── docs/         # 文档（结题报告 + 算法说明）
├── Data/         # 原始数据（不入 git，本地管理）
├── Output/       # 算法产物（不入 git，可重新生成）
└── charts/       # 图表（不入 git）
```

## 测试与脚本

```bash
# 单元测试
pytest

# scripts/ 下的脚本用 -m 运行（保证能 import 根级模块）
python -m scripts.generate_v4_results         # 生成 V4 可视化用 CSV
python -m scripts.eval_yield                  # 案例匹配良率评估
python -m scripts.dz_four_segment_validation  # 4 段分类标签验证
```

## 数据说明

`Data/` 包含汇总数据（2025-11-26 到 2025-12-01）及各方向（DZ/DY/BZ/BY）的单向数据。汇总数据与单向 DZ 数据格式不同，分别用对应脚本处理。详见各 `utils/README*.md`。

> 参考：`X9600Rail Binning Algorithm Flow v1.2.251119.pdf`（算法流程权威说明）。
