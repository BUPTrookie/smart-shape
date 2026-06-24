"""
DZ四段算法V4版本 - 物理意义驱动的第三段分类
段1: P1-P4 (端点差值法)
段2: P5-P8 (端点法直线度拟合)
段3: P9-P16 (物理意义分类：ARC_UP/ARC_DOWN/FLAT/WAVE)
段4: P17-P20 (端点差值法)

第三段基于物理特征的4类分类：
- ARC_UP（上圆弧）：中段单峰上凸，整体上升趋势
- ARC_DOWN（下圆弧）：中段单峰下凹，整体下降趋势
- FLAT（平缓型）：几乎无变形，标准差小
- WAVE（剧烈波动）：前后趋势剧变，离散程度大
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SegmentConfig:
    """段配置信息"""
    start_point: int
    end_point: int
    num_points: int
    method: str  # 'endpoint_diff' 或 'straightness_fit' 或 'physical_classification'
    threshold: float


class RailBinningCoreV4:
    """DZ四段算法V4版本核心类"""

    def __init__(self, rail_type: str = 'X9600_DZ'):
        self.rail_type = rail_type
        self.df = None
        self.df_processed = None
        self.mmm_threshold = 0.018

        # V4版本段配置 - 4段，但段3使用物理分类
        self.segments = {
            1: SegmentConfig(1, 4, 4, 'endpoint_diff', 0),     # P1-P4, 端点差值法 (调整阈值)
            2: SegmentConfig(5, 8, 4, 'straightness_fit', 0),  # P5-P8, 端点法直线度拟合 (调整阈值)
            3: SegmentConfig(9, 16, 8, 'physical_classification', 0.0),  # P9-P16, 物理意义分类 (不适用)
            4: SegmentConfig(17, 20, 4, 'endpoint_diff', 0),  # P17-P20, 端点差值法 (调整阈值)
        }

        logger.info(f"初始化 {rail_type} DZ四段算法V4版本处理器")

    def update_thresholds(self, thresholds: List[float]):
        """更新各段阈值"""
        if len(thresholds) != 4:
            raise ValueError(f"需要4个阈值，但提供了 {len(thresholds)} 个")

        for i, threshold in enumerate(thresholds, 1):
            self.segments[i].threshold = threshold

        logger.info(f"阈值已更新为: {thresholds}")

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载Excel数据"""
        try:
            self.df = pd.read_excel(file_path)
            logger.info(f"成功加载数据: {len(self.df)} 条记录")
            return self.df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def preprocess_data(self) -> pd.DataFrame:
        """数据预处理"""
        if self.df is None:
            raise ValueError("请先加载数据")

        logger.info("开始数据预处理")

        # 复制数据
        self.df_processed = self.df.copy()

        # 计算P1-P14最小二乘法拟合值
        self.df_processed = self._calculate_least_squares_fit(self.df_processed)

        logger.info(f"数据预处理完成，共处理 {len(self.df_processed)} 条记录")
        return self.df_processed

    def _calculate_least_squares_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算P1-P14最小二乘法拟合值"""
        logger.info("开始计算P1-P14最小二乘法拟合值")

        # 获取P1-P14列
        p_columns = [f'P{i}' for i in range(1, 15)]  # P1-P14
        x_values = np.arange(1, 15)  # x = 1, 2, ..., 14

        for idx, row in df.iterrows():
            y_values = row[p_columns].values

            # 去除NaN值进行拟合
            valid_indices = ~np.isnan(y_values)
            if np.sum(valid_indices) >= 2:
                x_valid = x_values[valid_indices]
                y_valid = y_values[valid_indices]

                # 最小二乘拟合 y = ax + b
                coeffs = np.polyfit(x_valid, y_valid, 1)
                a, b = coeffs

                # 计算拟合值
                fit_values = a * x_values + b
                lsd_fit_value = np.sqrt(np.mean((y_valid - (a * x_valid + b))**2))
            else:
                fit_values = np.full(14, np.nan)
                lsd_fit_value = np.nan

            # 将拟合值添加到DataFrame
            for i, fit_val in enumerate(fit_values):
                df.at[idx, f'P{i+1}_fit'] = fit_val

            df.at[idx, 'lsd_fit_value'] = lsd_fit_value

        return df

    def _calculate_endpoint_diff_feature(self, df: pd.DataFrame, segment_idx: int) -> pd.Series:
        """计算端点差值法特征值"""
        config = self.segments[segment_idx]
        start_col = f'P{config.start_point}'
        end_col = f'P{config.end_point}'

        return df[end_col] - df[start_col]

    def _calculate_straightness_fit_feature(self, df: pd.DataFrame, segment_idx: int) -> pd.Series:
        """计算端点法直线度拟合特征值"""
        config = self.segments[segment_idx]

        features = []

        for idx, row in df.iterrows():
            deviations = []

            try:
                # 计算端点差值
                start_point = f'P{config.start_point}'
                end_point = f'P{config.end_point}'
                endpoint_diff = row[end_point] - row[start_point]

                # 计算各点的理论值和偏差
                for i in range(config.num_points):
                    current_point = f'P{config.start_point + i}'
                    theoretical_value = row[start_point] + i * (endpoint_diff / (config.num_points - 1))
                    actual_value = row[current_point]

                    if pd.notna(actual_value) and pd.notna(theoretical_value):
                        deviation = actual_value - theoretical_value
                        deviations.append(deviation)

                # 选择绝对值最大的偏差作为特征值
                if deviations:
                    feature_value = max(deviations, key=lambda x: abs(x))
                else:
                    feature_value = np.nan

            except Exception as e:
                logger.warning(f"计算第{idx}行第{segment_idx}段特征值时出错: {e}")
                feature_value = np.nan

            features.append(feature_value)

        return pd.Series(features, index=df.index)

    def _calculate_physical_classification_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算第三段物理意义分类特征"""
        logger.info("开始第三段物理意义分类")

        # 计算三个物理特征
        classifications = []
        feature_details = []

        for idx, row in df.iterrows():
            try:
                # 获取P9-P16的数值
                p_values = row[[f'P{i}' for i in range(9, 17)]].values

                if pd.isna(p_values).any():
                    classifications.append('UNKNOWN')
                    feature_details.append({
                        'trend': np.nan,
                        'std_dev': np.nan,
                        'slope_left': np.nan,
                        'slope_right': np.nan,
                        'slope_diff': np.nan
                    })
                    continue

                # 特征1：整体趋势 (P9到P17的斜率)
                trend = (row['P17'] - row['P9']) / 8

                # 特征2：段内标准差 (P9-P17的离散程度)
                std_dev = np.std(p_values)

                # 特征3：前后半段斜率差 (WAVE专用)
                slope_left = (row['P13'] - row['P9']) / 4  # P9-P13斜率
                slope_right = (row['P17'] - row['P13']) / 4  # P13-P17斜率
                slope_diff = abs(slope_right - slope_left)

                # # 分类逻辑

                # if trend > 0.0008:
                #     category = 'ARC_UP'  # 上圆弧
                # elif trend < -0.008:
                #     category = 'ARC_DOWN'  # 下圆弧
                # elif std_dev < 0.02:
                #     category = 'FLAT'  # 平缓型
                # elif std_dev >= 0.06 and slope_diff > 0.08:
                #     category = 'WAVE'  # 剧烈波动
                # else:
                #     category = 'FLAT'  # 其他情况归为平缓

                # # 特殊处理：确保WAVE分类有足够的数据支持
                # if category == 'FLAT' and std_dev >= 0.08:
                #     if slope_diff > 0.1:
                #         category = 'WAVE'
                # 新的分类逻辑：优先级1：WAVE（剧烈波动型）- 最优先识别
                if std_dev < 0.03:  # ← 严格但范围小，捕获BIN18
                    category = 'FLAT'

                # 规则2：趋势明确 → ARC（趋势优先于波动）
                elif trend > 0.015:  # ← 阈值从0.008提升至0.015，避免轻微波动误判
                    category = 'ARC_UP'
                elif trend < -0.015:
                    category = 'ARC_DOWN'

                # 规则3：剧烈波动（无明确趋势）→ WAVE
                elif std_dev >= 0.08:  # ← 阈值提升至0.08，减少误伤
                    category = 'WAVE'

                # 规则4：中等波动+弱趋势 → 倾向ARC（波动比圆弧更需关注）
                else:
                    # 使用slope_diff辅助判断：转折陡峭 → WAVE，否则 → ARC
                    category = 'WAVE' if slope_diff > 0.10 else ('ARC_UP' if trend >= 0 else 'ARC_DOWN')


                classifications.append(category)
                feature_details.append({
                    'trend': trend,
                    'std_dev': std_dev,
                    'slope_left': slope_left,
                    'slope_right': slope_right,
                    'slope_diff': slope_diff
                })

            except Exception as e:
                logger.warning(f"计算第{idx}行第三段分类时出错: {e}")
                classifications.append('UNKNOWN')
                feature_details.append({
                    'trend': np.nan,
                    'std_dev': np.nan,
                    'slope_left': np.nan,
                    'slope_right': np.nan,
                    'slope_diff': np.nan
                })

        # 添加到DataFrame
        self.df_processed['seg3_category'] = classifications

        # 添加特征详细信息
        for key in ['trend', 'std_dev', 'slope_left', 'slope_right', 'slope_diff']:
            values = [details[key] for details in feature_details]
            self.df_processed[f'seg3_{key}'] = values

        logger.info(f"第三段物理分类完成: {pd.Series(classifications).value_counts().to_dict()}")
        return self.df_processed

    def calculate_features(self) -> pd.DataFrame:
        """计算所有段的特征值"""
        if self.df_processed is None:
            raise ValueError("请先进行数据预处理")

        logger.info("开始计算4段特征值")

        # 计算各段特征值
        for segment_idx in range(1, 5):
            config = self.segments[segment_idx]

            if config.method == 'endpoint_diff':
                feature_values = self._calculate_endpoint_diff_feature(self.df_processed, segment_idx)
            elif config.method == 'straightness_fit':
                feature_values = self._calculate_straightness_fit_feature(self.df_processed, segment_idx)
            elif config.method == 'physical_classification':
                # 第三段特殊处理
                if segment_idx == 3:
                    self._calculate_physical_classification_feature(self.df_processed)
                    continue
                else:
                    raise ValueError("物理分类仅适用于段3")
            else:
                raise ValueError(f"未知的计算方法: {config.method}")

            if segment_idx != 3:  # 段3已经特殊处理
                self.df_processed[f'segment_{segment_idx}_feature'] = feature_values

        logger.info("4段特征值计算完成")
        return self.df_processed

    def classify_segments(self) -> pd.DataFrame:
        """对各段进行分类"""
        if self.df_processed is None:
            raise ValueError("请先计算特征值")

        logger.info("开始各段分类")

        # 对前2段和第4段进行P/N分类
        shape_parts = []
        for segment_idx in [1, 2, 4]:  # 跳过段3，它使用物理分类
            feature_col = f'segment_{segment_idx}_feature'
            threshold = self.segments[segment_idx].threshold

            # P/N分类：特征值 >= 阈值则为P，否则为N
            labels = []
            for feature_value in self.df_processed[feature_col]:
                if pd.isna(feature_value):
                    labels.append('U')  # Unknown for NaN values
                else:
                    label = 'P' if feature_value >= threshold else 'N'
                    labels.append(label)

            shape_parts.append(labels)
            self.df_processed[f'segment_{segment_idx}_label'] = labels

        # 对段3进行物理分类标签映射
        seg3_labels = []
        physical_to_label = {
            'ARC_UP': 'A',      # 上圆弧 -> A
            'ARC_DOWN': 'R',     # 下圆弧 -> R (代表凹)
            'FLAT': 'F',         # 平缓型 -> F
            'WAVE': 'W',         # 剧烈波动 -> W
            'UNKNOWN': 'U'
        }

        for category in self.df_processed['seg3_category']:
            seg3_labels.append(physical_to_label.get(category, 'U'))

        self.df_processed['segment_3_label'] = seg3_labels
        shape_parts.insert(2, seg3_labels)  # 插入到第3个位置

        # 组合成4段Shape (P/N + 物理分类)
        shapes = []
        for i in range(len(self.df_processed)):
            shape = ''.join([shape_parts[j][i] for j in range(4)])
            shapes.append(shape)

        self.df_processed['shape'] = shapes
        logger.info("各段分类完成")

        return self.df_processed

    def apply_mmm_rules(self) -> pd.DataFrame:
        """应用MMM标签规则"""
        if self.df_processed is None:
            raise ValueError("请先进行段分类")

        logger.info("应用最小二乘拟合MMM标签规则")

        # 应用MMM规则
        final_shapes = []
        for idx, row in self.df_processed.iterrows():
            lsd_fit_value = row.get('lsd_fit_value', np.nan)

            if pd.notna(lsd_fit_value) and lsd_fit_value < self.mmm_threshold:
                # 前两个标签改为MMM，保持后两个标签
                # 注意：段3现在是物理分类标签(A/R/F/W/U)
                shape = 'MM' + row['shape'][2:]
            else:
                shape = row['shape']

            final_shapes.append(shape)

        self.df_processed['final_shape'] = final_shapes
        logger.info("MMM规则应用完成")

        return self.df_processed

    def classify_bins(self) -> pd.DataFrame:
        """基于Shape模式进行BIN分类"""
        if self.df_processed is None:
            raise ValueError("请先应用MMM规则")

        logger.info("基于Shape模式进行BIN分类")

        # 基于最终的Shape进行BIN分类
        bins = []
        for idx, shape in enumerate(self.df_processed['final_shape']):
            if shape.startswith('MM'):
                # MM开头的Shape
                if len(shape) >= 4 and shape[2] in ['P', 'N', 'F', 'W', 'U']:
                    if shape[2] == 'P' or shape[2] == 'A':  # P或A都算Pass
                        bin_name = 'BIN17'  # MMM + Pass
                    else:  # N, R, F, W, U都算Fail
                        bin_name = 'BIN18'  # MMM + Fail
                else:
                    bin_name = 'UNKNOWN'
            else:
                # 检查整体值决定是否合格
                p_values = []
                for i in range(1, 21):
                    p_val = self.df_processed.iloc[idx].get(f'P{i}')
                    if pd.notna(p_val):
                        p_values.append(p_val)

                if p_values:
                    max_val = max(p_values)
                    min_val = min(p_values)
                    if max_val - min_val <= 0.1:
                        bin_name = 'BINOK'
                    else:
                        bin_name = 'BIN100'
                else:
                    bin_name = 'UNKNOWN'

            bins.append(bin_name)

        self.df_processed['BIN'] = bins
        logger.info("BIN分类完成")

        return self.df_processed

    def process(self, df: pd.DataFrame = None, file_path: str = None) -> pd.DataFrame:
        """完整处理流程"""
        try:
            # 加载数据
            if df is not None:
                self.df = df.copy()
            elif file_path is not None:
                self.load_data(file_path)
            elif self.df is None:
                raise ValueError("需要提供数据或文件路径")

            logger.info(f"开始处理 {self.rail_type} 数据，共 {len(self.df)} 条记录")

            # 数据预处理
            self.df_processed = self._calculate_least_squares_fit(self.df.copy())

            # 计算各段特征值（含段3物理分类 FLAT/ARC_UP/ARC_DOWN/WAVE）
            self.calculate_features()

            # 复用已有正确分类链（审查 #2：原 process 用硬编码兜底，全 BINOK，BIN1-16 产不出）
            self.classify_segments()    # 段1/2/4 P/N + 段3物理标签 → shape
            self.apply_mmm_rules()      # MMM规则 → final_shape
            self.classify_bins()        # 基于整体值+MMM → BIN(BINOK/BIN100/BIN17/18/UNKNOWN)

            logger.info(f"处理完成，生成 {len(self.df_processed)} 条完整记录")
            return self.df_processed

        except Exception as e:
            logger.error(f"处理过程出错: {e}")
            raise

    def get_statistics(self) -> Dict:
        """获取分类统计信息"""
        if self.df_processed is None:
            return {}

        stats = {
            'total_records': len(self.df_processed),
            'bin_distribution': self.df_processed['BIN'].value_counts().to_dict(),
            'shape_distribution': self.df_processed.get('final_shape', pd.Series()).value_counts().to_dict(),
            'seg3_physical_distribution': self.df_processed.get('seg3_category', pd.Series()).value_counts().to_dict(),
            'mmm_count': len(self.df_processed[self.df_processed['lsd_fit_value'] < self.mmm_threshold]),
            'thresholds': [self.segments[i].threshold for i in range(1, 5)]
        }

        return stats

    def get_detailed_statistics(self) -> Dict:
        """获取详细统计信息，包含物理特征"""
        if self.df_processed is None:
            return {}

        # 物理分类统计
        seg3_stats = {}
        if 'seg3_trend' in self.df_processed.columns:
            seg3_stats['trend_stats'] = {
                'mean': self.df_processed['seg3_trend'].mean(),
                'std': self.df_processed['seg3_trend'].std(),
                'min': self.df_processed['seg3_trend'].min(),
                'max': self.df_processed['seg3_trend'].max()
            }

        if 'seg3_std_dev' in self.df_processed.columns:
            seg3_stats['std_dev_stats'] = {
                'mean': self.df_processed['seg3_std_dev'].mean(),
                'std': self.df_processed['seg3_std_dev'].std(),
                'min': self.df_processed['seg3_std_dev'].min(),
                'max': self.df_processed['seg3_std_dev'].max()
            }

        # 按物理分类分组的统计
        physical_groups = self.df_processed.groupby('seg3_category')
        seg3_stats['by_category'] = {}
        for category, group in physical_groups:
            if category != 'UNKNOWN':
                seg3_stats['by_category'][category] = {
                    'count': len(group),
                    'avg_trend': group['seg3_trend'].mean() if 'seg3_trend' in group.columns else np.nan,
                    'avg_std': group['seg3_std_dev'].mean() if 'seg3_std_dev' in group.columns else np.nan,
                }

        return seg3_stats


def main():
    """测试函数"""
    try:
        # 初始化处理器
        processor = RailBinningCoreV4('X9600_DZ')

        # 更新阈值 [段1, 段2, 段3, 段4]
        processor.update_thresholds([0, 0, 0, 0])  # 段3使用物理分类，阈值设为0

        # 处理数据
        result = processor.process(file_path="Data/total_final_processed.xlsx")

        # 获取统计信息
        stats = processor.get_statistics()
        detailed_stats = processor.get_detailed_statistics()

        print("DZ四段算法V4版本处理结果:")
        print(f"总记录数: {stats['total_records']}")
        print(f"BIN分布: {stats['bin_distribution']}")
        print(f"第三段物理分类分布: {stats['seg3_physical_distribution']}")
        print(f"MMM产品数: {stats['mmm_count']}")
        print(f"当前阈值: {stats['thresholds']}")

        print("\\n第三段详细统计:")
        for category, info in detailed_stats.get('by_category', {}).items():
            if category != 'UNKNOWN':
                print(f"  {category}: {info['count']} 条, 平均趋势: {info['avg_trend']:.3f}, 平均标准差: {info['avg_std']:.3f}")

        return result

    except Exception as e:
        print(f"测试失败: {e}")
        return None


if __name__ == "__main__":
    result = main()
