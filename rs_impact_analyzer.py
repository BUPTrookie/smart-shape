"""
整形压头影响量化分析系统
====================

功能：从历史整形数据中学习压头对测量点的影响规律

主要功能模块：
1. 数据读取与Pre/Post配对
2. 特征工程（位置特征、Pre曲线特征、交互项）
3. 多输出Ridge回归建模
4. 模型评估与影响系数导出
5. 仿真与可视化

使用方法：
    from rs_impact_analyzer import RSImpactAnalyzer
    analyzer = RSImpactAnalyzer()
    analyzer.run_full_pipeline()
"""

import pandas as pd
import numpy as np
import json
import logging
import warnings
import os
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# sklearn相关
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 导入配置
import rs_impact_config as config

warnings.filterwarnings("ignore")


class LoggerSetup:
    """日志配置工具"""

    @staticmethod
    def setup_logger(
        name: str, log_file: Optional[str] = None, level: str = "INFO"
    ) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # 清除现有处理器
        logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        if log_file and config.SAVE_LOG_TO_FILE:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
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
            self.logger.info(f"数据列: {list(df.columns)}")
            return df
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise

    def pair_pre_post(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将Pre和Post数据配对成整形样本

        输出格式：
        - pre_P1..pre_P20: 整形前测量值
        - post_P1..post_P20: 整形后测量值
        - delta_P1..delta_P20: 变化量
        - RS1X..RS4X, RS1Z..RS4Z: 压头参数
        """
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

            # 取第一条记录（通常一个工件只有一条Pre/Post）
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
                **{f"pre_{col}": pre_row[col] for col in config.POINT_COLUMNS},
                **{f"post_{col}": post_row[col] for col in config.POINT_COLUMNS},
            }

            # 计算delta
            for col in config.POINT_COLUMNS:
                sample[f"delta_{col}"] = post_row[col] - pre_row[col]

            # 添加压头参数
            for rs_name, rs_cols in config.RS_COLUMNS.items():
                sample[rs_cols["X"]] = pre_row[rs_cols["X"]]
                sample[rs_cols["Z"]] = pre_row[rs_cols["Z"]]

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
            # RSZ缺失值填充为0（未使用压头）
            df[rs_cols["Z"]] = df[rs_cols["Z"]].fillna(0)

            # RSX缺失值保持为None或NaN
            # df[rs_cols['X']] = df[rs_cols['X']]  # 保持原样

        # 异常值检测
        for rs_name, rs_cols in config.RS_COLUMNS.items():
            rsz_col = rs_cols["Z"]

            # 检测极端RSZ值
            extreme_mask = (df[rsz_col] < config.RSZ_MIN_THRESHOLD) | (
                df[rsz_col] > config.RSZ_MAX_THRESHOLD
            )

            if extreme_mask.any():
                extreme_count = extreme_mask.sum()
                self.logger.warning(
                    f"{rs_name}Z 发现 {extreme_count} 个极端值（范围外）"
                )
                # 可选：过滤掉这些样本
                # df = df[~extreme_mask]

        self.logger.info(f"清洗后数据量: {len(df)} 条")
        return df


class FeatureEngineer:
    """特征工程模块"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.rsx_positions = {}  # 存储每个压头的RSX档位集合
        self.feature_names = []  # 存储最终特征名列表
        self.pre_feature_names = []  # Pre曲线特征名
        self.interaction_feature_names = []  # 交互项特征名

    def extract_rsx_positions(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        提取每个压头的RSX离散档位集合

        优先级：
        1. 从RSX统计文件读取
        2. 从主数据中自动提取
        """
        self.logger.info("提取RSX档位集合...")

        # 尝试从统计文件读取
        if config.RSX_STATS_FILE and os.path.exists(config.RSX_STATS_FILE):
            try:
                stats_df = pd.read_csv(config.RSX_STATS_FILE)
                self.logger.info(f"从统计文件读取RSX档位: {config.RSX_STATS_FILE}")
                # 解析统计文件...（根据实际格式调整）
            except Exception as e:
                self.logger.warning(f"读取统计文件失败: {e}，将从主数据提取")

        # 从主数据提取
        for rs_name, rs_cols in config.RS_COLUMNS.items():
            rsx_col = rs_cols["X"]

            # 统计每个RSX值的出现频次
            rsx_values = df[rsx_col].dropna()
            value_counts = rsx_values.value_counts()

            # 过滤低频档位
            valid_positions = value_counts[
                value_counts >= config.MIN_POS_FREQUENCY
            ].index.tolist()

            self.rsx_positions[rs_name] = valid_positions
            self.logger.info(
                f"{rs_name}X 档位 ({len(valid_positions)} 个): {valid_positions}"
            )

        return self.rsx_positions

    def create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        为每个压头和位置构造特征：X_{h,pos} = RS_hZ * 1[RS_hX == pos]

        例如：
        - RS1_pos_21.5 = RS1Z * (RS1X == 21.5)
        - RS2_pos_0.0 = RS2Z * (RS2X == 0.0)
        """
        self.logger.info("构造位置特征...")

        feature_df = pd.DataFrame(index=df.index)

        for rs_name, rs_cols in config.RS_COLUMNS.items():
            rsx_col = rs_cols["X"]
            rsz_col = rs_cols["Z"]
            positions = self.rsx_positions[rs_name]

            for pos in positions:
                # 构造特征名
                feature_name = f"{rs_name}_pos_{pos:.4f}"
                self.feature_names.append(feature_name)

                # 构造特征：RSZ * 1[RSX == pos]
                feature_df[feature_name] = df[rsz_col] * (df[rsx_col] == pos).astype(
                    float
                )

        self.logger.info(f"位置特征数: {len(self.feature_names)}")
        return feature_df

    def create_pre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构造Pre曲线的全局特征（可选）

        特征包括：
        - 均值、标准差
        - 斜率（首尾差值）
        - 前/中/后段均值
        - 拟合误差
        """
        if not config.USE_PRE_FEATURES:
            self.logger.info("跳过Pre曲线特征（配置关闭）")
            return pd.DataFrame(index=df.index)

        self.logger.info("构造Pre曲线特征...")

        feature_df = pd.DataFrame(index=df.index)
        pre_cols = [f"pre_{col}" for col in config.POINT_COLUMNS]

        # 1. 全局统计特征
        feature_df["pre_mean"] = df[pre_cols].mean(axis=1)
        feature_df["pre_std"] = df[pre_cols].std(axis=1)
        self.pre_feature_names.extend(["pre_mean", "pre_std"])

        # 2. 斜率特征（首尾差值）
        feature_df["pre_slope"] = (
            df[f"pre_{config.POINT_COLUMNS[-1]}"] - df[f"pre_{config.POINT_COLUMNS[0]}"]
        )
        self.pre_feature_names.append("pre_slope")

        # 3. 前/中/后段均值
        segment1 = [f"pre_{col}" for col in config.POINT_COLUMNS[:4]]  # P1-P4
        segment2 = [f"pre_{col}" for col in config.POINT_COLUMNS[4:8]]  # P5-P8
        segment3 = [f"pre_{col}" for col in config.POINT_COLUMNS[8:16]]  # P9-P16
        segment4 = [f"pre_{col}" for col in config.POINT_COLUMNS[16:]]  # P17-P20

        feature_df["pre_seg1_mean"] = df[segment1].mean(axis=1)
        feature_df["pre_seg2_mean"] = df[segment2].mean(axis=1)
        feature_df["pre_seg3_mean"] = df[segment3].mean(axis=1)
        feature_df["pre_seg4_mean"] = df[segment4].mean(axis=1)
        self.pre_feature_names.extend(
            ["pre_seg1_mean", "pre_seg2_mean", "pre_seg3_mean", "pre_seg4_mean"]
        )

        self.logger.info(f"Pre曲线特征数: {len(self.pre_feature_names)}")
        return feature_df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构造压头交互项特征（可选）

        仅对Top-K高频压头组合构造：RSZ1 * RSZ2
        """
        if not config.USE_INTERACTION_FEATURES:
            self.logger.info("跳过交互项特征（配置关闭）")
            return pd.DataFrame(index=df.index)

        self.logger.info("构造交互项特征...")

        feature_df = pd.DataFrame(index=df.index)

        # 统计压头组合频次
        combo_counts = Counter()
        for _, row in df.iterrows():
            active_presses = []
            for rs_name, rs_cols in config.RS_COLUMNS.items():
                rsz_val = row[rs_cols["Z"]]
                if rsz_val != 0:  # 压头被使用
                    active_presses.append(rs_name)

            # 记录所有两两组合
            for i in range(len(active_presses)):
                for j in range(i + 1, len(active_presses)):
                    combo = tuple(sorted([active_presses[i], active_presses[j]]))
                    combo_counts[combo] += 1

        # 选择Top-K高频组合
        top_combos = combo_counts.most_common(config.TOP_K_INTERACTIONS)
        self.logger.info(f"Top-{config.TOP_K_INTERACTIONS}高频压头组合:")
        for combo, count in top_combos:
            self.logger.info(f"  {combo}: {count} 次")

        # 构造交互特征
        for combo, _ in top_combos:
            rs1, rs2 = combo
            feature_name = f"inter_{rs1}_{rs2}"
            self.interaction_feature_names.append(feature_name)

            rsz1_col = config.RS_COLUMNS[rs1]["Z"]
            rsz2_col = config.RS_COLUMNS[rs2]["Z"]

            feature_df[feature_name] = df[rsz1_col] * df[rsz2_col]

        self.logger.info(f"交互项特征数: {len(self.interaction_feature_names)}")
        return feature_df

    def build_feature_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        构造完整的特征矩阵和目标变量

        返回：
        - X: 特征矩阵
        - y: 目标变量（delta_P1..delta_P20）
        """
        self.logger.info("构造特征矩阵...")

        # 提取RSX档位
        self.extract_rsx_positions(df)

        # 各类特征
        pos_features = self.create_position_features(df)
        pre_features = self.create_pre_features(df)
        interaction_features = self.create_interaction_features(df)

        # 合并所有特征
        X = pd.concat([pos_features, pre_features, interaction_features], axis=1)

        # 填充缺失值（未使用的压头特征为0）
        X = X.fillna(0)

        # 目标变量
        target_cols = [f"delta_{col}" for col in config.POINT_COLUMNS]
        y = df[target_cols].copy()

        self.logger.info(f"特征矩阵形状: X={X.shape}, y={y.shape}")
        self.logger.info(
            f"总特征数: {len(self.feature_names) + len(self.pre_feature_names) + len(self.interaction_feature_names)}"
        )

        return X, y


class ModelTrainer:
    """模型训练与评估"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.model = None
        self.scaler = StandardScaler()
        self.best_alpha = None
        self.feature_names = []
        self.cv_scores = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.25,
        random_state: int = None,
    ) -> Tuple:
        """
        划分训练集和测试集

        参数:
            test_size: 测试集比例，默认0.15（即15%）
            random_state: 随机种子

        返回:
            X_train, X_test, y_train, y_test
        """
        if random_state is None:
            random_state = config.RANDOM_STATE

        self.logger.info(f"划分训练集/测试集: 测试集比例={test_size*100}%")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.logger.info(f"  训练集: {len(X_train)} 条 ({(1-test_size)*100:.1f}%)")
        self.logger.info(f"  测试集: {len(X_test)} 条 ({test_size*100:.1f}%)")

        return X_train, X_test, y_train, y_test

    def train_with_cv(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, feature_names: List[str]
    ) -> Dict:
        """
        使用K折交叉验证选择最佳Ridge模型，仅在训练集上训练

        参数:
            X_train: 训练集特征
            y_train: 训练集目标变量
            feature_names: 特征名列表

        返回: 训练集评估指标
        """
        self.logger.info("开始模型训练与交叉验证...")
        self.feature_names = feature_names

        # 标准化特征（仅用训练集fit）
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=X_train.columns, index=X_train.index
        )

        # K折交叉验证选择alpha
        self.logger.info(f"交叉验证: {config.N_FOLDS}折")
        self.logger.info(f"Alpha候选: {config.ALPHA_CANDIDATES}")

        best_score = -np.inf
        best_alpha = None

        # 使用第一个点位作为示例选择alpha（实际应用中可用平均得分）
        target_point = "delta_P1"
        y_target = y_train[target_point]

        for alpha in config.ALPHA_CANDIDATES:
            kf = KFold(
                n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
            )
            model = MultiOutputRegressor(
                Ridge(alpha=alpha, random_state=config.RANDOM_STATE)
            )

            # 交叉验证（仅在训练集上）
            scores = cross_val_score(
                model, X_train_scaled, y_train, cv=kf, scoring="r2", n_jobs=-1
            )
            avg_score = scores.mean()

            self.logger.info(
                f"Alpha={alpha}: 平均R2={avg_score:.4f} (std={scores.std():.4f})"
            )

            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha

        self.best_alpha = best_alpha
        self.logger.info(f"最佳Alpha: {best_alpha} (R2={best_score:.4f})")

        # 用最佳alpha训练最终模型（仅在训练集上）
        self.model = MultiOutputRegressor(
            Ridge(alpha=best_alpha, random_state=config.RANDOM_STATE)
        )
        self.model.fit(X_train_scaled, y_train)

        self.logger.info("模型训练完成")

        # 评估训练集性能
        metrics = self.evaluate_model(X_train_scaled, y_train, dataset_name="训练集")
        return metrics

    def evaluate_on_test(self) -> Dict:
        """
        在测试集上评估模型性能

        返回: 测试集评估指标
        """
        if self.X_test is None or self.y_test is None:
            self.logger.warning("测试集不存在，跳过测试集评估")
            return {}

        self.logger.info("评估测试集性能...")

        # 标准化测试集特征（使用训练集的scaler）
        X_test_scaled = self.scaler.transform(self.X_test)
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=self.X_test.columns, index=self.X_test.index
        )

        # 评估
        metrics = self.evaluate_model(X_test_scaled, self.y_test, dataset_name="测试集")
        return metrics

    def evaluate_model(
        self, X: pd.DataFrame, y: pd.DataFrame, dataset_name: str = "数据集"
    ) -> Dict:
        """
        评估模型性能

        参数:
            X: 特征矩阵
            y: 真实目标变量
            dataset_name: 数据集名称（训练集/测试集）

        返回: 评估指标字典
        """
        self.logger.info(f"评估{dataset_name}性能...")

        # 预测
        y_pred = self.model.predict(X)

        # 计算每个点位的指标
        metrics = {}
        r2_scores = []
        rmse_scores = []

        for i, col in enumerate(y.columns):
            y_true = y.iloc[:, i].values
            y_pred_col = y_pred[:, i]

            r2 = r2_score(y_true, y_pred_col)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_col))

            metrics[col] = {"R2": r2, "RMSE": rmse}
            r2_scores.append(r2)
            rmse_scores.append(rmse)

        # 全局指标
        metrics["global"] = {
            "mean_R2": np.mean(r2_scores),
            "std_R2": np.std(r2_scores),
            "mean_RMSE": np.mean(rmse_scores),
            "weighted_R2": np.average(r2_scores),  # 可根据样本量加权
        }

        self.logger.info(
            f"{dataset_name} - 平均R2: {metrics['global']['mean_R2']:.4f} ± {metrics['global']['std_R2']:.4f}"
        )
        self.logger.info(
            f"{dataset_name} - 平均RMSE: {metrics['global']['mean_RMSE']:.4f}"
        )

        return metrics

    def extract_influence_coefficients(self, feature_names: List[str]) -> Dict:
        """
        提取影响系数字典

        格式：influence[h][pos] = {P1: coef, ..., P20: coef}
        """
        self.logger.info("提取影响系数...")

        influence = {}
        intercept = {}

        # 获取每个输出点位的系数
        for i, point_col in enumerate(config.POINT_COLUMNS):
            target_name = f"delta_{point_col}"
            estimator = self.model.estimators_[i]  # 第i个Ridge模型

            # 截距
            intercept[target_name] = float(estimator.intercept_)

            # 系数
            coefs = estimator.coef_

            # 解析特征名，提取压头和位置信息
            for j, feat_name in enumerate(feature_names):
                if feat_name.startswith("RS") and "_pos_" in feat_name:
                    # 位置特征：RS1_pos_21.5000
                    parts = feat_name.split("_pos_")
                    rs_name = parts[0]  # RS1
                    pos = float(parts[1])  # 21.5

                    if rs_name not in influence:
                        influence[rs_name] = {}

                    if pos not in influence[rs_name]:
                        influence[rs_name][pos] = {}

                    influence[rs_name][pos][target_name] = float(coefs[j])

        self.logger.info("影响系数提取完成")
        return influence, intercept


class ResultExporter:
    """结果导出模块"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def export_influence_coefficients(
        self,
        influence: Dict,
        intercept: Dict,
        feature_names: List[str],
        output_path: str,
    ):
        """导出影响系数到JSON"""
        self.logger.info(f"导出影响系数到: {output_path}")

        result = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "model_type": "MultiOutputRidge",
                "n_points": len(config.POINT_COLUMNS),
                "n_features": len(feature_names),
                "feature_names": feature_names,
            },
            "intercept": intercept,
            "influence_coefficients": influence,
        }

        # 格式化JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        self.logger.info("影响系数导出完成")

    def export_metrics(self, metrics: Dict, output_path: str):
        """导出模型评估指标"""
        self.logger.info(f"导出评估指标到: {output_path}")

        # 转换为可序列化格式
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                metrics_serializable[key] = {k: float(v) for k, v in value.items()}
            else:
                metrics_serializable[key] = float(value)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)

        self.logger.info("评估指标导出完成")

    def export_predictions(
        self,
        y_true: pd.DataFrame,
        y_pred: np.ndarray,
        df_original: pd.DataFrame,
        output_path: str,
        dataset_name: str = "预测",
    ):
        """
        导出预测结果

        参数:
            y_true: 真实值
            y_pred: 预测值
            df_original: 原始数据（用于获取ID）
            output_path: 输出路径
            dataset_name: 数据集名称（训练集/测试集）
        """
        self.logger.info(f"导出{dataset_name}预测结果到: {output_path}")

        result_df = df_original[[config.ID_COLUMN]].copy()

        # 添加真实值和预测值
        for i, col in enumerate(config.POINT_COLUMNS):
            target_name = f"delta_{col}"
            result_df[f"{target_name}_true"] = y_true[target_name].values
            result_df[f"{target_name}_pred"] = y_pred[:, i]
            result_df[f"{target_name}_error"] = (
                result_df[f"{target_name}_true"] - result_df[f"{target_name}_pred"]
            )

        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"{dataset_name}预测结果导出完成")

    def export_training_data(self, X: pd.DataFrame, y: pd.DataFrame, output_path: str):
        """导出训练数据"""
        self.logger.info(f"导出训练数据到: {output_path}")

        train_df = pd.concat([X, y], axis=1)
        train_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        self.logger.info("训练数据导出完成")


class RSImpactAnalyzer:
    """整形压头影响量化分析主类"""

    def __init__(self):
        # 配置日志
        self.logger = LoggerSetup.setup_logger(
            "RSImpactAnalyzer", config.LOG_FILE_PATH, config.LOG_LEVEL
        )

        # 初始化各模块
        self.data_loader = DataLoader(self.logger)
        self.feature_engineer = FeatureEngineer(self.logger)
        self.trainer = ModelTrainer(self.logger)
        self.exporter = ResultExporter(self.logger)

        # 数据存储
        self.raw_data = None
        self.paired_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_data = None  # 训练集原始数据
        self.test_data = None  # 测试集原始数据
        self.metrics_train = None  # 训练集指标
        self.metrics_test = None  # 测试集指标
        self.influence = None
        self.intercept = None

    def run_full_pipeline(self, test_size: float = 0.15):
        """
        运行完整分析流程

        参数:
            test_size: 测试集比例，默认0.15（即15%）
        """
        self.logger.info("=" * 80)
        self.logger.info("整形压头影响量化分析系统启动")
        self.logger.info("=" * 80)

        try:
            # 1. 数据加载与配对
            self.logger.info("\n[1/7] 数据加载与Pre/Post配对")
            self.raw_data = self.data_loader.load_raw_data()
            self.paired_data = self.data_loader.pair_pre_post(self.raw_data)
            self.paired_data = self.data_loader.clean_and_validate(self.paired_data)

            # 2. 特征工程
            self.logger.info("\n[2/7] 特征工程")
            self.X, self.y = self.feature_engineer.build_feature_matrix(
                self.paired_data
            )

            # 导出完整特征数据
            if config.EXPORT_TRAIN_DATA:
                self.exporter.export_training_data(
                    self.X, self.y, config.TRAIN_DATA_CSV_PATH
                )

            # 3. 划分训练集/测试集
            self.logger.info(
                f"\n[3/7] 划分训练集/测试集 (测试集比例: {test_size*100}%)"
            )
            X_train, X_test, y_train, y_test = self.trainer.split_train_test(
                self.X, self.y, test_size=test_size
            )
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            # 划分原始数据（用于导出预测结果）
            train_indices = X_train.index
            test_indices = X_test.index
            self.train_data = self.paired_data.loc[train_indices].reset_index(drop=True)
            self.test_data = self.paired_data.loc[test_indices].reset_index(drop=True)

            # 数据分布诊断
            self.logger.info(f"\n数据分布诊断:")

            # 1. 压头使用情况分布
            for rs_name in ["RS1", "RS2", "RS3", "RS4"]:
                rsz_col = f"{rs_name}Z"

                # 训练集: 使用该压头的比例
                train_used = (
                    (self.train_data[rsz_col] > 0).sum() / len(self.train_data) * 100
                )

                # 测试集: 使用该压头的比例
                test_used = (
                    (self.test_data[rsz_col] > 0).sum() / len(self.test_data) * 100
                )

                diff = abs(train_used - test_used)
                status = "✓" if diff < 5 else "⚠️"

                self.logger.info(
                    f"  {status} {rs_name}使用率 - 训练集:{train_used:.1f}%, 测试集:{test_used:.1f}% (差异:{diff:.1f}%)"
                )

            # 2. Delta值统计
            delta_cols = [f"delta_{col}" for col in config.POINT_COLUMNS]
            train_delta_std = self.train_data[delta_cols].values.std()
            test_delta_std = self.test_data[delta_cols].values.std()

            self.logger.info(f"\nDelta值标准差:")
            self.logger.info(f"  训练集: {train_delta_std:.6f}")
            self.logger.info(f"  测试集: {test_delta_std:.6f}")

            if train_delta_std > test_delta_std * 1.2:
                self.logger.warning(f"  ⚠️ 训练集波动显著更大,可能包含更多离群值")

            # 4. 模型训练（仅用训练集）
            self.logger.info("\n[4/7] 模型训练与交叉验证（训练集）")
            feature_names = (
                self.feature_engineer.feature_names
                + self.feature_engineer.pre_feature_names
                + self.feature_engineer.interaction_feature_names
            )
            self.metrics_train = self.trainer.train_with_cv(
                X_train, y_train, feature_names
            )

            # 5. 测试集评估
            self.logger.info("\n[5/7] 测试集评估")
            self.metrics_test = self.trainer.evaluate_on_test()

            # 6. 提取影响系数
            self.logger.info("\n[6/7] 提取影响系数")
            self.influence, self.intercept = (
                self.trainer.extract_influence_coefficients(feature_names)
            )

            # 7. 导出结果
            self.logger.info("\n[7/7] 导出结果")

            # 导出影响系数
            self.exporter.export_influence_coefficients(
                self.influence,
                self.intercept,
                feature_names,
                config.INFLUENCE_JSON_PATH,
            )

            # 导出训练集和测试集指标
            if config.EXPORT_METRICS:
                # 训练集指标
                train_metrics_path = config.METRICS_JSON_PATH.replace(
                    ".json", "_train.json"
                )
                self.exporter.export_metrics(self.metrics_train, train_metrics_path)

                # 测试集指标
                if self.metrics_test:
                    test_metrics_path = config.METRICS_JSON_PATH.replace(
                        ".json", "_test.json"
                    )
                    self.exporter.export_metrics(self.metrics_test, test_metrics_path)

            # 导出预测结果
            if config.EXPORT_PREDICTIONS:
                # 训练集预测
                y_train_pred = self.trainer.model.predict(
                    self.trainer.scaler.transform(X_train)
                )
                train_predictions_path = config.PREDICTIONS_CSV_PATH.replace(
                    ".csv", "_train.csv"
                )
                self.exporter.export_predictions(
                    y_train,
                    y_train_pred,
                    self.train_data,
                    train_predictions_path,
                    "训练集",
                )

                # 测试集预测
                if self.metrics_test:
                    y_test_pred = self.trainer.model.predict(
                        self.trainer.scaler.transform(X_test)
                    )
                    test_predictions_path = config.PREDICTIONS_CSV_PATH.replace(
                        ".csv", "_test.csv"
                    )
                    self.exporter.export_predictions(
                        y_test,
                        y_test_pred,
                        self.test_data,
                        test_predictions_path,
                        "测试集",
                    )

            # 8. 生成报告
            self.logger.info("\n[8/8] 生成分析报告")
            self.print_summary_report()

            self.logger.info("\n" + "=" * 80)
            self.logger.info("分析完成！")
            self.logger.info(f"结果保存在: {config.OUTPUT_DIR}")
            self.logger.info("=" * 80)

        except Exception as e:
            self.logger.error(f"分析过程出错: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            raise

    def print_summary_report(self):
        """打印汇总报告"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("分析汇总报告")
        self.logger.info("=" * 80)

        # 数据统计
        self.logger.info(f"\n数据统计:")
        self.logger.info(f"  原始记录: {len(self.raw_data)} 条")
        self.logger.info(f"  配对样本: {len(self.paired_data)} 对")

        # 训练集和测试集统计
        if self.X_train is not None and self.X_test is not None:
            train_pct = len(self.X_train) / len(self.X) * 100
            test_pct = len(self.X_test) / len(self.X) * 100
            self.logger.info(f"  训练集: {len(self.X_train)} 条 ({train_pct:.1f}%)")
            self.logger.info(f"  测试集: {len(self.X_test)} 条 ({test_pct:.1f}%)")

        # 模型性能 - 训练集
        if self.metrics_train:
            self.logger.info(f"\n模型性能 - 训练集:")
            self.logger.info(f"  最佳Alpha: {self.trainer.best_alpha}")
            self.logger.info(
                f"  平均R2: {self.metrics_train['global']['mean_R2']:.4f} ± {self.metrics_train['global']['std_R2']:.4f}"
            )
            self.logger.info(
                f"  平均RMSE: {self.metrics_train['global']['mean_RMSE']:.4f}"
            )

            # Top点位性能
            self.logger.info(f"\n训练集点位R2 Top 5:")
            point_r2 = [
                (col, self.metrics_train[f"delta_{col}"]["R2"])
                for col in config.POINT_COLUMNS
                if f"delta_{col}" in self.metrics_train
            ]
            point_r2_sorted = sorted(point_r2, key=lambda x: x[1], reverse=True)[:5]
            for point, r2 in point_r2_sorted:
                self.logger.info(f"  {point}: {r2:.4f}")

        # 模型性能 - 测试集
        if self.metrics_test:
            self.logger.info(f"\n模型性能 - 测试集:")
            self.logger.info(
                f"  平均R2: {self.metrics_test['global']['mean_R2']:.4f} ± {self.metrics_test['global']['std_R2']:.4f}"
            )
            self.logger.info(
                f"  平均RMSE: {self.metrics_test['global']['mean_RMSE']:.4f}"
            )

            # Top点位性能
            self.logger.info(f"\n测试集点位R2 Top 5:")
            point_r2 = [
                (col, self.metrics_test[f"delta_{col}"]["R2"])
                for col in config.POINT_COLUMNS
                if f"delta_{col}" in self.metrics_test
            ]
            point_r2_sorted = sorted(point_r2, key=lambda x: x[1], reverse=True)[:5]
            for point, r2 in point_r2_sorted:
                self.logger.info(f"  {point}: {r2:.4f}")

            # 过拟合检测
            if self.metrics_train:
                train_r2 = self.metrics_train["global"]["mean_R2"]
                test_r2 = self.metrics_test["global"]["mean_R2"]
                overfitting = train_r2 - test_r2
                self.logger.info(f"\n过拟合检测:")
                self.logger.info(f"  R2差异 (训练-测试): {overfitting:.4f}")

                if overfitting > 0.05:
                    # 训练集显著高于测试集 -> 典型过拟合
                    self.logger.warning(
                        f"  ⚠️ 可能存在过拟合 (训练R² >> 测试R², 差异 > 0.05)"
                    )
                elif overfitting < -0.05:
                    # 测试集显著高于训练集 -> 数据分布不均或训练集有离群值
                    self.logger.warning(
                        f"  ⚠️ 异常: 测试集性能显著优于训练集 (差异 < -0.05)"
                    )
                    self.logger.warning(f"  可能原因:")
                    self.logger.warning(f"    1. 随机划分导致训练集包含更多难预测样本")
                    self.logger.warning(f"    2. 训练集存在离群值,拉低了整体R²")
                    self.logger.warning(f"    3. 建议: 使用分层采样或交叉验证")
                else:
                    # 差异在±0.05之间 -> 正常范围
                    self.logger.info(f"  ✓ 模型泛化性能良好 (差异在±0.05范围内)")

        # Top5最大偏差分析
        self._print_top_errors()

        # 影响系数概览
        if self.influence:
            self.logger.info(f"\n影响系数概览:")
            for rs_name in sorted(self.influence.keys()):
                positions = sorted(self.influence[rs_name].keys())
                self.logger.info(f"  {rs_name}: {len(positions)} 个位置档位")

    def _print_top_errors(self):
        """输出训练集和测试集的Top5最大偏差样本"""
        self.logger.info(f"\n" + "=" * 80)
        self.logger.info("Top5 最大偏差样本分析")
        self.logger.info("=" * 80)

        # 读取预测结果文件
        try:
            train_pred_path = config.PREDICTIONS_CSV_PATH.replace(".csv", "_train.csv")
            test_pred_path = config.PREDICTIONS_CSV_PATH.replace(".csv", "_test.csv")

            if os.path.exists(train_pred_path):
                df_train = pd.read_csv(train_pred_path)
                self._analyze_top_errors(df_train, "训练集")

            if os.path.exists(test_pred_path):
                df_test = pd.read_csv(test_pred_path)
                self._analyze_top_errors(df_test, "测试集")

        except Exception as e:
            self.logger.warning(f"无法分析最大偏差: {e}")

    def _analyze_top_errors(self, df: pd.DataFrame, dataset_name: str):
        """分析单个数据集的Top5最大偏差"""
        self.logger.info(f"\n{dataset_name} - Top5 最大偏差样本:")

        # 获取所有误差列
        error_cols = [col for col in df.columns if col.endswith("_error")]

        # 计算每个样本的总绝对误差
        df["total_abs_error"] = df[error_cols].abs().sum(axis=1)

        # 获取最大误差的5个样本
        top5 = df.nlargest(5, "total_abs_error")

        for idx, row in top5.iterrows():
            barcode = row[config.ID_COLUMN]
            total_error = row["total_abs_error"]

            # 找到该样本误差最大的点位
            point_errors = [(col, abs(row[col])) for col in error_cols]
            max_point, max_error = max(point_errors, key=lambda x: x[1])

            # 提取点位名称
            point_name = max_point.replace("_error", "")

            # 获取真实值和预测值
            true_col = max_point.replace("_error", "_true")
            pred_col = max_point.replace("_error", "_pred")
            true_val = row[true_col]
            pred_val = row[pred_col]

            self.logger.info(f"  #{idx+1} 条码: {barcode}")
            self.logger.info(f"      总误差: {total_error:.4f}")
            self.logger.info(f"      最大误差点位: {point_name} (误差={max_error:.4f})")
            self.logger.info(f"      真实值: {true_val:.6f}, 预测值: {pred_val:.6f}")


def main():
    """主函数示例"""
    print("=" * 80)
    print("整形压头影响量化分析系统")
    print("=" * 80)
    print("\n功能：从历史数据学习压头对测量点的影响规律")
    print(f"输入数据: {config.INPUT_DATA_PATH}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print("\n开始运行...\n")

    # 创建分析器并运行
    analyzer = RSImpactAnalyzer()
    analyzer.run_full_pipeline(test_size=0.2)  # 25%测试集,75%训练集

    print("\n程序执行完毕！请查看日志和输出文件。")


if __name__ == "__main__":
    main()
