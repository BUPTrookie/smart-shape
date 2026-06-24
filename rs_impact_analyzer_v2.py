"""
整形压头影响量化分析系统 - RMSE优化版
======================================

功能：从历史整形数据中学习压头对测量点的影响规律，并进行RMSE优化

主要优化（暂定）：
1. 距离核特征：基于物理距离的衰减核特征
2. 工况模式分模型：按压头使用模式分组建模
3. 异常值处理：识别和处理离群样本
4. ElasticNet支持：可选的L1+L2正则化

使用方法：
    from rs_impact_analyzer_v2 import RSImpactAnalyzerV2
    analyzer = RSImpactAnalyzerV2()
    analyzer.run_optimization_pipeline()
"""

import pandas as pd
import numpy as np
import json
import logging
import warnings
import os
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Union
from itertools import product

# sklearn相关
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 导入配置
import rs_impact_config as config

warnings.filterwarnings('ignore')


class LoggerSetup:
    """日志配置工具"""

    @staticmethod
    def setup_logger(name: str, log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # 清除现有处理器
        logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        if log_file and config.SAVE_LOG_TO_FILE:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger


class DataLoader:
    """数据加载与Pre/Post配对"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load_raw_data(self) -> pd.DataFrame:
        """加载Excel原始数据"""
        self.logger.info(f"正在加载数据文件: {config.INPUT_DATA_PATH}")
        try:
            df = pd.read_excel(config.INPUT_DATA_PATH, sheet_name=config.SHEET_NAME)
            self.logger.info(f"成功加载 {len(df)} 条记录")
            self.logger.info(f"数据列: {list(df.columns)[:20]}...")
            return df
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise

    def pair_pre_post(self, df: pd.DataFrame) -> pd.DataFrame:
        """将Pre和Post数据配对成整形样本"""
        self.logger.info("开始Pre/Post配对...")

        # 分离Pre和Post数据
        pre_df = df[df[config.STATUS_COLUMN] == config.PRE_STATUS_VALUE].copy()
        post_df = df[df[config.STATUS_COLUMN] == config.POST_STATUS_VALUE].copy()

        self.logger.info(f"Pre记录: {len(pre_df)} 条, Post记录: {len(post_df)} 条")

        # 配对数据
        paired_samples = []
        unmatched_ids = []

        for barcode in pre_df[config.ID_COLUMN].unique():
            pre_row = pre_df[pre_df[config.ID_COLUMN] == barcode]
            post_row = post_df[post_df[config.ID_COLUMN] == barcode]

            if len(pre_row) == 0 or len(post_row) == 0:
                unmatched_ids.append(barcode)
                continue

            # 取第一条记录
            pre_row = pre_row.iloc[0]
            post_row = post_row.iloc[0]

            # 检查测量点完整性
            pre_points = pre_row[config.POINT_COLUMNS]
            post_points = post_row[config.POINT_COLUMNS]

            if pre_points.isna().any() or post_points.isna().any():
                self.logger.warning(f"工件 {barcode} 存在缺失测量点，已跳过")
                continue

            # 构造配对样本
            sample = {
                config.ID_COLUMN: barcode,
                **{f'pre_{col}': pre_row[col] for col in config.POINT_COLUMNS},
                **{f'post_{col}': post_row[col] for col in config.POINT_COLUMNS},
            }

            # 计算delta
            for col in config.POINT_COLUMNS:
                sample[f'delta_{col}'] = post_row[col] - pre_row[col]

            # 添加压头参数
            for rs_name, rs_cols in config.RS_COLUMNS.items():
                sample[rs_cols['X']] = pre_row[rs_cols['X']]
                sample[rs_cols['Z']] = pre_row[rs_cols['Z']]

            paired_samples.append(sample)

        paired_df = pd.DataFrame(paired_samples)
        self.logger.info(f"成功配对 {len(paired_df)} 对样本")
        if unmatched_ids:
            self.logger.warning(f"未配对的工件数: {len(unmatched_ids)}")

        return paired_df

    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗与异常值检测"""
        self.logger.info("开始数据清洗与验证...")

        # 处理缺失压头参数
        for rs_name, rs_cols in config.RS_COLUMNS.items():
            df[rs_cols['Z']] = df[rs_cols['Z']].fillna(0)

        # 异常值检测
        for rs_name, rs_cols in config.RS_COLUMNS.items():
            rsz_col = rs_cols['Z']
            extreme_mask = (df[rsz_col] < config.RSZ_MIN_THRESHOLD) | \
                          (df[rsz_col] > config.RSZ_MAX_THRESHOLD)

            if extreme_mask.any():
                extreme_count = extreme_mask.sum()
                self.logger.warning(f"{rs_name}Z 发现 {extreme_count} 个极端值（范围外）")

        self.logger.info(f"清洗后数据量: {len(df)} 条")
        return df


class OutlierHandler:
    """异常值处理模块"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.outlier_stats = {}

    def analyze_delta_distribution(self, df: pd.DataFrame) -> Dict:
        """分析每个点位delta的分布"""
        self.logger.info("分析delta分布...")

        target_cols = [f'delta_{col}' for col in config.POINT_COLUMNS]
        stats = {}

        for col in target_cols:
            values = df[col].values
            stats[col] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'q0.5': np.percentile(values, 0.5),
                'q99.5': np.percentile(values, 99.5),
                'q1': np.percentile(values, 1),
                'q99': np.percentile(values, 99),
            }

        self.outlier_stats = stats
        return stats

    def handle_outliers(self, df: pd.DataFrame, method: str = 'clip',
                       clip_percentile: List[float] = [0.5, 99.5],
                       zscore_threshold: float = 3.0) -> Tuple[pd.DataFrame, Dict]:
        """处理异常值"""
        if not config.OUTLIER_CONFIG['enabled']:
            self.logger.info("异常值处理未启用")
            return df, {}

        self.logger.info(f"处理异常值 (方法: {method})...")

        df_clean = df.copy()
        target_cols = [f'delta_{col}' for col in config.POINT_COLUMNS]
        handling_info = {}

        for col in target_cols:
            if method == 'clip':
                # 截断到指定分位数
                lower = np.percentile(df[col], clip_percentile[0])
                upper = np.percentile(df[col], clip_percentile[1])

                n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
                df_clean[col] = df[col].clip(lower=lower, upper=upper)

                handling_info[col] = {'n_clipped': int(n_clipped), 'lower': lower, 'upper': upper}

            elif method == 'remove':
                # 剔除Z-score超过阈值的样本
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > zscore_threshold
                n_outliers = outlier_mask.sum()

                if n_outliers > 0:
                    self.logger.warning(f"{col}: 发现 {n_outliers} 个Z-score异常值")

                handling_info[col] = {'n_outliers': int(n_outliers)}

        self.logger.info(f"异常值处理完成")
        return df_clean, handling_info


class KernelFeatureEngineer:
    """距离核特征工程模块"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.kernel_feature_names = []
        self.point_coords = config.POINT_X_COORDS

    def create_kernel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建基于距离衰减的核特征"""
        if not config.OPTIMIZATION_CONFIG['use_kernel_features']:
            self.logger.info("距离核特征未启用")
            return pd.DataFrame(index=df.index)

        self.logger.info("构造距离核特征...")

        feature_df = pd.DataFrame(index=df.index)
        lambdas = config.KERNEL_FEATURES_CONFIG['lambdas']
        segments = config.KERNEL_FEATURES_CONFIG['segments']
        use_projection = config.KERNEL_FEATURES_CONFIG['use_projection']

        for rs_name, rs_cols in config.RS_COLUMNS.items():
            rsx_col = rs_cols['X']
            rsz_col = rs_cols['Z']

            # 对每个压头，获取RSX和RSZ
            rsx_values = df[rsx_col].values
            rsz_values = df[rsz_col].values

            # 对每个尺度lambda
            for lambda_val in lambdas:
                if use_projection:
                    # 方案1: 投影到各分段
                    for seg_idx, segment in enumerate(segments):
                        seg_name = f'seg{seg_idx+1}'

                        # 计算该段内每个点的核函数值
                        segment_features = []
                        for point_idx in segment:
                            point_name = f'P{point_idx}'
                            point_x = self.point_coords[point_name]

                            # 计算距离 d = |RSX - X_Pj|
                            distances = np.abs(rsx_values - point_x)

                            # 核函数: RSZ * exp(-d/lambda)
                            kernel_values = rsz_values * np.exp(-distances / lambda_val)

                            segment_features.append(kernel_values)

                        # 投影: 求和
                        projection = np.sum(segment_features, axis=0)
                        feature_name = f'{rs_name}_kernel_lambda{lambda_val:.0f}_{seg_name}'

                        feature_df[feature_name] = projection
                        self.kernel_feature_names.append(feature_name)

                else:
                    # 方案2: 为每个点单独构造特征（维度更高）
                    for point_idx in range(1, 21):
                        point_name = f'P{point_idx}'
                        point_x = self.point_coords[point_name]

                        distances = np.abs(rsx_values - point_x)
                        kernel_values = rsz_values * np.exp(-distances / lambda_val)

                        feature_name = f'{rs_name}_kernel_lambda{lambda_val:.0f}_P{point_idx}'
                        feature_df[feature_name] = kernel_values
                        self.kernel_feature_names.append(feature_name)

        # 填充缺失值（未使用的压头特征为0）
        feature_df = feature_df.fillna(0)

        self.logger.info(f"距离核特征数: {len(self.kernel_feature_names)}")
        return feature_df


class FeatureEngineer:
    """特征工程模块（整合原有特征和新特征）"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.rsx_positions = {}
        self.feature_names = []
        self.pre_feature_names = []
        self.interaction_feature_names = []

        # 初始化核特征工程师
        self.kernel_engineer = KernelFeatureEngineer(logger)

    def extract_rsx_positions(self, df: pd.DataFrame) -> Dict[str, List]:
        """提取每个压头的RSX离散档位集合"""
        self.logger.info("提取RSX档位集合...")

        # 从主数据提取
        for rs_name, rs_cols in config.RS_COLUMNS.items():
            rsx_col = rs_cols['X']
            rsx_values = df[rsx_col].dropna()
            value_counts = rsx_values.value_counts()
            valid_positions = value_counts[value_counts >= config.MIN_POS_FREQUENCY].index.tolist()

            self.rsx_positions[rs_name] = valid_positions
            self.logger.info(f"{rs_name}X 档位 ({len(valid_positions)} 个): {valid_positions}")

        return self.rsx_positions

    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构造位置特征（原有特征，可选）"""
        self.logger.info("构造位置特征...")

        feature_df = pd.DataFrame(index=df.index)

        for rs_name, rs_cols in config.RS_COLUMNS.items():
            rsx_col = rs_cols['X']
            rsz_col = rs_cols['Z']
            positions = self.rsx_positions[rs_name]

            for pos in positions:
                feature_name = f'{rs_name}_pos_{pos:.4f}'
                self.feature_names.append(feature_name)
                feature_df[feature_name] = df[rsz_col] * (df[rsx_col] == pos).astype(float)

        self.logger.info(f"位置特征数: {len(self.feature_names)}")
        return feature_df

    def create_pre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构造Pre曲线特征"""
        if not config.USE_PRE_FEATURES:
            return pd.DataFrame(index=df.index)

        self.logger.info("构造Pre曲线特征...")

        feature_df = pd.DataFrame(index=df.index)
        pre_cols = [f'pre_{col}' for col in config.POINT_COLUMNS]

        # 全局统计特征
        feature_df['pre_mean'] = df[pre_cols].mean(axis=1)
        feature_df['pre_std'] = df[pre_cols].std(axis=1)
        self.pre_feature_names.extend(['pre_mean', 'pre_std'])

        # 斜率特征
        feature_df['pre_slope'] = df[f'pre_{config.POINT_COLUMNS[-1]}'] - df[f'pre_{config.POINT_COLUMNS[0]}']
        self.pre_feature_names.append('pre_slope')

        # 分段均值
        segment1 = [f'pre_{col}' for col in config.POINT_COLUMNS[:4]]
        segment2 = [f'pre_{col}' for col in config.POINT_COLUMNS[4:8]]
        segment3 = [f'pre_{col}' for col in config.POINT_COLUMNS[8:16]]
        segment4 = [f'pre_{col}' for col in config.POINT_COLUMNS[16:]]

        feature_df['pre_seg1_mean'] = df[segment1].mean(axis=1)
        feature_df['pre_seg2_mean'] = df[segment2].mean(axis=1)
        feature_df['pre_seg3_mean'] = df[segment3].mean(axis=1)
        feature_df['pre_seg4_mean'] = df[segment4].mean(axis=1)
        self.pre_feature_names.extend(['pre_seg1_mean', 'pre_seg2_mean', 'pre_seg3_mean', 'pre_seg4_mean'])

        self.logger.info(f"Pre曲线特征数: {len(self.pre_feature_names)}")
        return feature_df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构造压头交互项特征"""
        if not config.USE_INTERACTION_FEATURES:
            return pd.DataFrame(index=df.index)

        self.logger.info("构造交互项特征...")

        feature_df = pd.DataFrame(index=df.index)

        # 统计压头组合频次
        combo_counts = Counter()
        for _, row in df.iterrows():
            active_presses = []
            for rs_name, rs_cols in config.RS_COLUMNS.items():
                if row[rs_cols['Z']] != 0:
                    active_presses.append(rs_name)

            for i in range(len(active_presses)):
                for j in range(i+1, len(active_presses)):
                    combo = tuple(sorted([active_presses[i], active_presses[j]]))
                    combo_counts[combo] += 1

        # 选择Top-K高频组合
        top_combos = combo_counts.most_common(config.TOP_K_INTERACTIONS)

        # 构造交互特征
        for combo, _ in top_combos:
            rs1, rs2 = combo
            feature_name = f'inter_{rs1}_{rs2}'
            self.interaction_feature_names.append(feature_name)

            rsz1_col = config.RS_COLUMNS[rs1]['Z']
            rsz2_col = config.RS_COLUMNS[rs2]['Z']

            feature_df[feature_name] = df[rsz1_col] * df[rsz2_col]

        self.logger.info(f"交互项特征数: {len(self.interaction_feature_names)}")
        return feature_df

    def build_feature_matrix(self, df: pd.DataFrame, use_kernel: bool = False,
                            use_position: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """构造完整的特征矩阵和目标变量"""
        self.logger.info("构造特征矩阵...")

        # 提取RSX档位
        self.extract_rsx_positions(df)

        feature_dfs = []

        # 1. 位置特征（可选，与核特征二选一）
        if use_position and not use_kernel:
            pos_features = self.create_position_features(df)
            feature_dfs.append(pos_features)

        # 2. 距离核特征（新增）
        if use_kernel:
            kernel_features = self.kernel_engineer.create_kernel_features(df)
            feature_dfs.append(kernel_features)

        # 3. Pre特征
        pre_features = self.create_pre_features(df)
        feature_dfs.append(pre_features)

        # 4. 交互特征
        interaction_features = self.create_interaction_features(df)
        feature_dfs.append(interaction_features)

        # 合并所有特征
        if feature_dfs:
            X = pd.concat(feature_dfs, axis=1)
        else:
            X = pd.DataFrame(index=df.index)

        X = X.fillna(0)

        # 目标变量
        target_cols = [f'delta_{col}' for col in config.POINT_COLUMNS]
        y = df[target_cols].copy()

        self.logger.info(f"特征矩阵形状: X={X.shape}, y={y.shape}")

        return X, y


class PatternModelManager:
    """工况模式分模型管理器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.patterns = {}
        self.pattern_models = {}
        self.global_model = None

    def create_pattern_key(self, row: pd.Series) -> Tuple:
        """为每个样本创建工况模式key"""
        active_mask = tuple([
            row[config.RS_COLUMNS[rs]['Z']] != 0
            for rs in config.RS_COLUMNS.keys()
        ])

        if config.PATTERN_MODEL_CONFIG['use_position_in_key']:
            # 包含RSX位置
            positions = tuple([
                row[config.RS_COLUMNS[rs]['X']] if active_mask[i] else None
                for i, rs in enumerate(config.RS_COLUMNS.keys())
            ])
            return (active_mask, positions)
        else:
            # 只用active_mask
            return active_mask

    def analyze_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析工况模式分布"""
        self.logger.info("分析工况模式...")

        # 为每个样本计算pattern key
        pattern_keys = []
        for _, row in df.iterrows():
            key = self.create_pattern_key(row)
            pattern_keys.append(key)

        df = df.copy()
        df['_pattern_key'] = pattern_keys

        # 统计每个pattern的出现次数
        pattern_counts = Counter(pattern_keys)
        self.logger.info(f"发现 {len(pattern_counts)} 种不同的工况模式")

        # 输出top模式
        top_patterns = pattern_counts.most_common(10)
        self.logger.info("Top-10工况模式:")
        for pattern, count in top_patterns:
            self.logger.info(f"  {pattern}: {count} 次")

        self.patterns = pattern_counts
        return df

    def train_pattern_models(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.DataFrame,
                             scaler: StandardScaler, model_class, model_params: Dict):
        """为每个工况模式训练独立模型"""
        if not config.OPTIMIZATION_CONFIG['use_pattern_models']:
            self.logger.info("工况模式分模型未启用")
            return

        self.logger.info("训练工况模式分模型...")

        # 确保df和X对齐
        df = df.reset_index(drop=True)
        X = X.reset_index(drop=True)

        min_samples = config.PATTERN_MODEL_CONFIG['min_samples_for_pattern']

        # 为高频pattern训练独立模型
        for pattern, count in self.patterns.items():
            if count >= min_samples:
                self.logger.info(f"为模式 {pattern} 训练独立模型 (样本数: {count})")

                # 获取该pattern的样本
                pattern_mask = df['_pattern_key'] == pattern
                X_pattern = X[pattern_mask]
                y_pattern = y[pattern_mask]

                # 训练模型
                model = MultiOutputRegressor(model_class(**model_params))
                model.fit(X_pattern, y_pattern)

                self.pattern_models[pattern] = {
                    'model': model,
                    'sample_count': count,
                    'mask': pattern_mask
                }

        self.logger.info(f"共为 {len(self.pattern_models)} 种模式训练了独立模型")

        # 训练global模型（用于低频pattern）
        self.logger.info("训练全局模型（用于低频模式）...")
        self.global_model = MultiOutputRegressor(model_class(**model_params))
        self.global_model.fit(X, y)

    def predict_with_pattern_models(self, X: pd.DataFrame) -> np.ndarray:
        """使用工况模式模型预测"""
        if not config.OPTIMIZATION_CONFIG['use_pattern_models']:
            return None

        # 简化实现：使用global模型预测
        # 实际应该根据每个样本的pattern选择对应模型
        return self.global_model.predict(X)


class ModelTrainer:
    """模型训练与评估"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.feature_names = []

    def select_best_model(self, X: pd.DataFrame, y: pd.DataFrame,
                         use_elasticnet: bool = False) -> Tuple:
        """选择最佳模型和参数"""
        self.logger.info("选择最佳模型参数...")

        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        if use_elasticnet:
            # ElasticNet网格搜索
            param_grid = {
                'estimator__alpha': config.ELASTICNET_CONFIG['alpha_candidates'],
                'estimator__l1_ratio': config.ELASTICNET_CONFIG['l1_ratio_candidates']
            }
            base_model = ElasticNet(random_state=config.RANDOM_STATE)
        else:
            # Ridge网格搜索
            param_grid = {
                'estimator__alpha': config.ALPHA_CANDIDATES
            }
            base_model = Ridge(random_state=config.RANDOM_STATE)

        model = MultiOutputRegressor(base_model)

        # K折交叉验证
        kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

        # 网格搜索
        grid_search = GridSearchCV(
            model, param_grid, cv=kf, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_scaled, y)

        self.logger.info(f"最佳参数: {grid_search.best_params_}")
        self.logger.info(f"最佳R2得分: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        return self.model, X_scaled, grid_search.best_score_

    def train_model(self, X: pd.DataFrame, y: pd.DataFrame,
                   use_elasticnet: bool = False) -> Dict:
        """训练模型"""
        self.logger.info("开始模型训练...")

        model, X_scaled, best_score = self.select_best_model(X, y, use_elasticnet)

        # 评估
        metrics = self.evaluate_model(X_scaled, y)

        return metrics, X_scaled

    def evaluate_model(self, X: pd.DataFrame, y: pd.DataFrame, use_cv: bool = True) -> Dict:
        """评估模型性能（使用交叉验证）"""
        self.logger.info("评估模型性能...")

        metrics = {}
        r2_scores = []
        rmse_scores = []

        if use_cv:
            # 使用交叉验证评估（更客观）
            self.logger.info("使用5折交叉验证评估...")
            kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

            for i, col in enumerate(y.columns):
                y_true = y.iloc[:, i].values

                # 对每个输出进行交叉验证
                cv_r2 = cross_val_score(self.model.estimators_[i], X, y_true,
                                       cv=kf, scoring='r2', n_jobs=-1)
                cv_rmse = -cross_val_score(self.model.estimators_[i], X, y_true,
                                          cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)

                r2 = np.mean(cv_r2)
                rmse = np.mean(cv_rmse)

                metrics[col] = {
                    'R2': r2,
                    'RMSE': rmse,
                    'R2_std': np.std(cv_r2),
                    'RMSE_std': np.std(cv_rmse)
                }
                r2_scores.append(r2)
                rmse_scores.append(rmse)

        else:
            # 在训练集上评估（可能过拟合）
            self.logger.warning("在训练集上评估，结果可能过于乐观")
            y_pred = self.model.predict(X)

            for i, col in enumerate(y.columns):
                y_true = y.iloc[:, i].values
                y_pred_col = y_pred[:, i]

                r2 = r2_score(y_true, y_pred_col)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred_col))

                metrics[col] = {'R2': r2, 'RMSE': rmse}
                r2_scores.append(r2)
                rmse_scores.append(rmse)

        # 全局指标
        metrics['global'] = {
            'mean_R2': np.mean(r2_scores),
            'std_R2': np.std(r2_scores),
            'mean_RMSE': np.mean(rmse_scores),
            'std_RMSE': np.std(rmse_scores),
        }

        # RMSE<=0.01的点位统计
        rmse_array = np.array(rmse_scores)
        n_le_01 = (rmse_array <= 0.01).sum()
        metrics['global']['n_points_with_RMSE_le_01'] = int(n_le_01)
        metrics['global']['pct_points_with_RMSE_le_01'] = float(n_le_01 / len(rmse_scores) * 100)

        self.logger.info(f"平均R2: {metrics['global']['mean_R2']:.4f} ± {metrics['global']['std_R2']:.4f}")
        self.logger.info(f"平均RMSE: {metrics['global']['mean_RMSE']:.6f} ± {metrics['global']['std_RMSE']:.6f}")
        self.logger.info(f"RMSE<=0.01的点位: {n_le_01}/20 ({n_le_01/len(rmse_scores)*100:.1f}%)")

        return metrics


class ResultExporter:
    """结果导出模块"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def export_metrics(self, metrics: Dict, output_path: str, model_type: str = "Baseline"):
        """导出模型评估指标"""
        self.logger.info(f"导出{model_type}评估指标到: {output_path}")

        # 转换为可序列化格式
        metrics_serializable = {
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
        }

        for key, value in metrics.items():
            if isinstance(value, dict):
                metrics_serializable[key] = {k: float(v) for k, v in value.items()}
            else:
                metrics_serializable[key] = float(value)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)

        self.logger.info("评估指标导出完成")

    def export_predictions(self, y_true: pd.DataFrame, y_pred: np.ndarray,
                          df_original: pd.DataFrame, output_path: str):
        """导出预测结果"""
        self.logger.info(f"导出预测结果到: {output_path}")

        result_df = df_original[[config.ID_COLUMN]].copy()

        # 添加真实值和预测值
        for i, col in enumerate(config.POINT_COLUMNS):
            target_name = f'delta_{col}'
            result_df[f'{target_name}_true'] = y_true[target_name].values
            result_df[f'{target_name}_pred'] = y_pred[:, i]
            result_df[f'{target_name}_error'] = result_df[f'{target_name}_true'] - result_df[f'{target_name}_pred']

        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        self.logger.info("预测结果导出完成")

    def export_patterns_summary(self, patterns: Dict, output_path: str):
        """导出工况模式汇总"""
        self.logger.info(f"导出工况模式汇总到: {output_path}")

        summary_data = []
        for pattern, count in patterns.items():
            summary_data.append({
                'pattern': str(pattern),
                'count': count,
                'is_modeled_separately': count >= config.PATTERN_MODEL_CONFIG['min_samples_for_pattern']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('count', ascending=False)
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        self.logger.info("工况模式汇总导出完成")


class RSImpactAnalyzerV2:
    """整形压头影响量化分析主类 - RMSE优化版"""

    def __init__(self):
        # 配置日志
        log_path = os.path.join(config.OUTPUT_DIR, "optimization_analysis.log")
        self.logger = LoggerSetup.setup_logger(
            'RSImpactAnalyzerV2', log_path, config.LOG_LEVEL
        )

        # 初始化各模块
        self.data_loader = DataLoader(self.logger)
        self.outlier_handler = OutlierHandler(self.logger)
        self.feature_engineer = FeatureEngineer(self.logger)
        self.pattern_manager = PatternModelManager(self.logger)
        self.trainer = ModelTrainer(self.logger)
        self.exporter = ResultExporter(self.logger)

        # 数据存储
        self.raw_data = None
        self.paired_data = None
        self.clean_data = None
        self.baseline_metrics = None
        self.improved_metrics = None

    def run_baseline(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """运行基线模型"""
        self.logger.info("\n" + "="*80)
        self.logger.info("[阶段1] 运行基线模型")
        self.logger.info("="*80)

        # 构造基线特征
        X, y = self.feature_engineer.build_feature_matrix(
            self.clean_data, use_kernel=False, use_position=True
        )

        # 训练基线模型
        metrics, X_scaled = self.trainer.train_model(X, y, use_elasticnet=False)

        # 预测
        y_pred = self.trainer.model.predict(X_scaled)

        # 导出结果
        self.exporter.export_metrics(metrics, config.BASELINE_METRICS_PATH, "Baseline")
        self.exporter.export_predictions(y, y_pred, self.clean_data, config.BASELINE_PREDICTIONS_PATH)

        self.baseline_metrics = metrics
        return X, y, metrics

    def run_optimized_model(self, X_baseline, y) -> Dict:
        """运行优化模型"""
        self.logger.info("\n" + "="*80)
        self.logger.info("[阶段2] 运行优化模型")
        self.logger.info("="*80)

        # 1. 加入距离核特征
        self.logger.info("\n[优化步骤1] 加入距离核特征")
        X, y = self.feature_engineer.build_feature_matrix(
            self.clean_data, use_kernel=True, use_position=False
        )

        # 训练
        metrics_kernel, X_scaled = self.trainer.train_model(X, y, use_elasticnet=False)
        self.logger.info(f"加入核特征后 - 平均RMSE: {metrics_kernel['global']['mean_RMSE']:.6f}")

        # 2. 可选：按工况模式分模型（暂时跳过，因为实现较复杂）
        # TODO: 实现工况模式分模型

        # 3. 可选：ElasticNet
        if config.OPTIMIZATION_CONFIG['use_elasticnet']:
            self.logger.info("\n[优化步骤3] 使用ElasticNet")
            metrics_en, _ = self.trainer.train_model(X, y, use_elasticnet=True)
            self.logger.info(f"ElasticNet - 平均RMSE: {metrics_en['global']['mean_RMSE']:.6f}")

            if metrics_en['global']['mean_RMSE'] < metrics_kernel['global']['mean_RMSE']:
                self.logger.info("ElasticNet表现更好，采用ElasticNet结果")
                metrics_kernel = metrics_en

        # 导出结果
        self.exporter.export_metrics(metrics_kernel, config.IMPROVED_METRICS_PATH, "Improved")

        y_pred = self.trainer.model.predict(X_scaled)
        self.exporter.export_predictions(y, y_pred, self.clean_data, config.IMPROVED_PREDICTIONS_PATH)

        self.improved_metrics = metrics_kernel
        return metrics_kernel

    def compare_results(self):
        """对比基线和优化结果"""
        self.logger.info("\n" + "="*80)
        self.logger.info("[阶段3] 结果对比")
        self.logger.info("="*80)

        baseline = self.baseline_metrics['global']
        improved = self.improved_metrics['global']

        self.logger.info(f"\n{'指标':<30} {'基线':>15} {'优化后':>15} {'改进':>15}")
        self.logger.info("-" * 80)

        self.logger.info(f"{'平均R2':<30} {baseline['mean_R2']:>15.4f} {improved['mean_R2']:>15.4f} {improved['mean_R2']-baseline['mean_R2']:>+14.4f}")
        self.logger.info(f"{'平均RMSE':<30} {baseline['mean_RMSE']:>15.6f} {improved['mean_RMSE']:>15.6f} {improved['mean_RMSE']-baseline['mean_RMSE']:>+14.6f}")
        self.logger.info(f"{'RMSE<=0.01点位数':<30} {baseline['n_points_with_RMSE_le_01']:>15d} {improved['n_points_with_RMSE_le_01']:>15d} {improved['n_points_with_RMSE_le_01']-baseline['n_points_with_RMSE_le_01']:>+14d}")

        # 各点位RMSE对比
        self.logger.info(f"\n各点位RMSE对比:")
        self.logger.info(f"{'点位':<10} {'基线RMSE':>12} {'优化RMSE':>12} {'改进':>12}")
        self.logger.info("-" * 50)

        for col in config.POINT_COLUMNS:
            delta_col = f'delta_{col}'
            baseline_rmse = self.baseline_metrics[delta_col]['RMSE']
            improved_rmse = self.improved_metrics[delta_col]['RMSE']
            improvement = baseline_rmse - improved_rmse

            marker = " ✓" if improved_rmse <= 0.01 else ""
            self.logger.info(f"{col:<10} {baseline_rmse:>12.6f} {improved_rmse:>12.6f} {improvement:>+11.6f}{marker}")

    def run_optimization_pipeline(self):
        """运行完整的优化流程"""
        self.logger.info("="*80)
        self.logger.info("整形压头影响量化分析系统 - RMSE优化版启动")
        self.logger.info("="*80)

        try:
            # 1. 数据加载与配对
            self.logger.info("\n[步骤1] 数据加载与Pre/Post配对")
            self.raw_data = self.data_loader.load_raw_data()
            self.paired_data = self.data_loader.pair_pre_post(self.raw_data)
            self.paired_data = self.data_loader.clean_and_validate(self.paired_data)

            # 2. 异常值分析
            self.logger.info("\n[步骤2] 异常值分析")
            self.outlier_handler.analyze_delta_distribution(self.paired_data)

            # 3. 异常值处理
            if config.OUTLIER_CONFIG['enabled']:
                self.logger.info("\n[步骤3] 异常值处理")
                self.clean_data, handling_info = self.outlier_handler.handle_outliers(
                    self.paired_data,
                    method=config.OUTLIER_CONFIG['method'],
                    clip_percentile=config.OUTLIER_CONFIG['clip_percentile']
                )
            else:
                self.clean_data = self.paired_data.copy()

            # 4. 工况模式分析（用于了解数据分布）
            self.logger.info("\n[步骤4] 工况模式分析")
            self.clean_data = self.pattern_manager.analyze_patterns(self.clean_data)

            # 导出工况模式汇总
            self.exporter.export_patterns_summary(
                self.pattern_manager.patterns, config.PATTERNS_SUMMARY_PATH
            )

            # 5. 运行基线
            X_baseline, y, baseline_metrics = self.run_baseline()

            # 6. 运行优化模型
            improved_metrics = self.run_optimized_model(X_baseline, y)

            # 7. 对比结果
            self.compare_results()

            self.logger.info("\n" + "="*80)
            self.logger.info("优化分析完成！")
            self.logger.info(f"结果保存在: {config.OUTPUT_DIR}")
            self.logger.info("="*80)

        except Exception as e:
            self.logger.error(f"分析过程出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


def main():
    """主函数"""
    print("="*80)
    print("整形压头影响量化分析系统 - RMSE优化版")
    print("="*80)
    print("\n优化策略:")
    print(f"  - 距离核特征: {config.OPTIMIZATION_CONFIG['use_kernel_features']}")
    print(f"  - 工况模式分模型: {config.OPTIMIZATION_CONFIG['use_pattern_models']}")
    print(f"  - 异常值处理: {config.OPTIMIZATION_CONFIG['use_outlier_handling']}")
    print(f"  - ElasticNet: {config.OPTIMIZATION_CONFIG['use_elasticnet']}")
    print(f"\n输入数据: {config.INPUT_DATA_PATH}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print("\n开始运行...\n")

    analyzer = RSImpactAnalyzerV2()
    analyzer.run_optimization_pipeline()

    print("\n程序执行完毕！请查看日志和输出文件。")


if __name__ == "__main__":
    main()
