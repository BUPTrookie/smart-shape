"""
整形压头影响量化 - 配置文件
============

使用说明：
1. 根据实际情况修改以下路径和参数
2. 主要配置项：
   - 数据路径配置
   - 列名配置
   - 特征工程开关
   - 模型参数
"""

import os

# ==================== 路径配置 ====================
# 输入数据路径（修改为你的实际路径）
INPUT_DATA_PATH = "Data/total_final_processed.xlsx"

# RSX统计文件路径（如果有的话，留空则从主数据自动提取）
RSX_STATS_FILE = "Output/RS_analysis/rs_summary.csv"  # 或者 "" 禁用

# 输出目录
OUTPUT_DIR = "Output/RS_impact_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 影响系数输出文件
INFLUENCE_JSON_PATH = os.path.join(OUTPUT_DIR, "influence_coefficients.json")

# ==================== 列名配置 ====================
# Excel工作表名称
SHEET_NAME = "Reshaping"

# 工件ID列名（用于Pre/Post配对）
ID_COLUMN = "Barcode"

# 测量点列名（P1-P20）
POINT_COLUMNS = [f'P{i}' for i in range(1, 21)]

# 压头相关列名
RS_COLUMNS = {
    'RS1': {'X': 'RS1X', 'Z': 'RS1Z'},
    'RS2': {'X': 'RS2X', 'Z': 'RS2Z'},
    'RS3': {'X': 'RS3X', 'Z': 'RS3Z'},
    'RS4': {'X': 'RS4X', 'Z': 'RS4Z'},
}

# 状态列（用于区分Pre/Post）
STATUS_COLUMN = "Status"

# Pre/Post标识值
PRE_STATUS_VALUE = "Pre"
POST_STATUS_VALUE = "Post"

# ==================== 特征工程配置 ====================
# RSX最小出现频次阈值（过滤低频档位）
MIN_POS_FREQUENCY = 5  # 出现次数少于5次的档位将被忽略

# 是否使用Pre曲线特征
USE_PRE_FEATURES = True  # True: 加入pre曲线的全局特征（均值、斜率等）

# 是否使用压头交互项
USE_INTERACTION_FEATURES = True  # True: 加入多压头同时作用的交互特征

# 交互项Top-K配置（仅对最频繁的K个压头组合构造交互）
TOP_K_INTERACTIONS = 10

# ==================== 模型配置 ====================
# 交叉验证折数
N_FOLDS = 5

# Ridge正则化参数候选集
ALPHA_CANDIDATES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

# 随机种子（保证可复现）
RANDOM_STATE = 42

# ==================== 异常值检测配置 ====================
# RSZ下压量的合理范围（超出则告警）
RSZ_MIN_THRESHOLD = -100.0  # 最小下压量
RSZ_MAX_THRESHOLD = 50.0    # 最大下压量

# ==================== 日志配置 ====================
# 日志级别（DEBUG/INFO/WARNING/ERROR）
LOG_LEVEL = "INFO"

# 是否保存详细日志到文件
SAVE_LOG_TO_FILE = True
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "analysis.log")

# ==================== 可视化配置 ====================
# 是否生成可视化图表
GENERATE_PLOTS = True

# 图表保存路径
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ==================== 导出配置 ====================
# 是否导出训练数据集
EXPORT_TRAIN_DATA = True
TRAIN_DATA_CSV_PATH = os.path.join(OUTPUT_DIR, "training_data.csv")

# 是否导出模型评估结果
EXPORT_METRICS = True
METRICS_JSON_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")

# 是否导出预测结果
EXPORT_PREDICTIONS = True
PREDICTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")

# ==================== RMSE优化配置 ====================
# 优化策略开关
OPTIMIZATION_CONFIG = {
    'use_kernel_features': False,     # 是否使用距离核特征（False=使用原有位置桶特征）
    'use_pattern_models': True,        # 是否按工况模式分模型
    'use_outlier_handling': True,      # 是否处理异常值
    'use_elasticnet': True,            # 是否使用ElasticNet（True: ElasticNet, False: Ridge）
}

# 点位X坐标（如果没有文件，使用等间距分布）
# 假设P1-P20均匀分布在-70到70的范围内
POINT_X_COORDS = {f'P{i}': -70.0 + (i-1) * (140.0/19.0) for i in range(1, 21)}

# 距离核特征配置
KERNEL_FEATURES_CONFIG = {
    'lambdas': [3.0, 7.0, 15.0, 30.0, 60.0],  # 优化后的距离衰减尺度（更细致的近场+更宽的远场）
    'segments': [                              # 分段定义（用于紧凑特征）
        [1, 2, 3, 4],      # P1-P4
        [5, 6, 7, 8],      # P5-P8
        [9, 10, 11, 12, 13, 14, 15, 16],  # P9-P16
        [17, 18, 19, 20],  # P17-P20
    ],
    'use_projection': True,  # True: 使用投影特征, False: 为每个点单独构造特征
}

# 工况模式分模型配置
PATTERN_MODEL_CONFIG = {
    'min_samples_for_pattern': 50,  # 最小样本数阈值
    'use_position_in_key': True,    # 是否将RSX位置包含在key中
}

# 异常值处理配置
OUTLIER_CONFIG = {
    'enabled': True,
    'method': 'clip',  # 'clip': 截断, 'remove': 剔除
    'clip_percentile': [0.5, 99.5],  # 截断分位数
    'zscore_threshold': 3.0,         # Z-score阈值（用于remove方法）
}

# ElasticNet配置（如果OPTIMIZATION_CONFIG['use_elasticnet']=True）
ELASTICNET_CONFIG = {
    'alpha_candidates': [0.001, 0.01, 0.1, 1.0],
    'l1_ratio_candidates': [0.1, 0.5, 0.7, 0.9],  # 0: Ridge, 1: Lasso
}

# 对比测试输出路径
BASELINE_METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_baseline.json")
IMPROVED_METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_improved.json")
BASELINE_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions_baseline.csv")
IMPROVED_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions_improved.csv")
PATTERNS_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "patterns_summary.csv")
