import pandas as pd
import numpy as np
import logging
from scipy.signal import find_peaks

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_v4_physical_classification():
    """测试V4版本的物理分类"""
    try:
        # 加载数据
        df = pd.read_excel("Data/total_final_processed.xlsx")
        logger.info(f"加载数据: {len(df)} 条记录")

        # 只处理Pre状态数据
        df_pre = df[df["Status"] == "Pre"].copy()
        logger.info(f"Pre状态数据: {len(df_pre)} 条记录")

        # MMM数据分类先舍弃
        # 初始化MMM阈值
        # mmm_threshold = 0.018

        # 计算P1-P14最小二乘拟合值
        # logger.info("计算P1-P14最小二乘拟合值...")
        # p_columns = [f"P{i}" for i in range(1, 15)]
        # x_values = np.arange(1, 15)

        # for idx, row in df_pre.iterrows():
        #     y_values = row[p_columns].values.astype(float)
        #     valid_indices = ~pd.isna(y_values)

        #     if np.sum(valid_indices) >= 2:
        #         x_valid = x_values[valid_indices]
        #         y_valid = y_values[valid_indices]
        #         coeffs = np.polyfit(x_valid, y_valid, 1)
        #         lsd_fit_value = np.sqrt(
        #             np.mean((y_valid - (coeffs[0] * x_valid + coeffs[1])) ** 2)
        #         )
        #     else:
        #         lsd_fit_value = np.nan

        #     df_pre.at[idx, "lsd_fit_value"] = lsd_fit_value

        # logger.info("最小二乘拟合值计算完成")

        # 注释掉MMM预分类部分
        # # 应用MMM预分类 是MMM的则标签为MMM_TYPE，否则为NORMAL_TYPE
        # bins = []
        # for idx, row in df_pre.iterrows():
        #     lsd_fit_value = row.get("lsd_fit_value", np.nan)
        #     if pd.notna(lsd_fit_value):
        #         if lsd_fit_value < mmm_threshold:
        #             bins.append("MMM_TYPE")
        #         else:
        #             bins.append("NORMAL_TYPE")
        #     else:
        #         bins.append("UNKNOWN")

        # df_pre["pre_bin"] = bins
        # # MMM数据分类先舍弃
        # # 注释掉MMM分类,让所有数据都流经分类逻辑
        # # # 只对非MMM数据进行物理分类
        # # normal_data = df_pre[df_pre['pre_bin'] == 'NORMAL_TYPE'].copy()
        # # logger.info(f"正常数据: {len(normal_data)} 条, MMM数据: {len(df_pre) - len(normal_data)} 条")

        # 使用所有数据进行分类
        normal_data = df_pre.copy()
        logger.info(f"处理所有数据: {len(normal_data)} 条")

        # 计算段1, 2, 4的特征
        # 段1: P1-P4 端点差值法
        normal_data["seg1_feature"] = normal_data["P4"] - normal_data["P1"]

        # 段2: P5-P8 端点法直线度拟合
        seg2_features = []
        for idx, row in normal_data.iterrows():
            start_val = row["P5"]
            end_val = row["P8"]
            endpoint_diff = end_val - start_val

            deviations = []
            for i in range(4):
                actual_val = row[f"P{5+i}"]
                theoretical_val = start_val + i * (endpoint_diff / 3)
                deviation = actual_val - theoretical_val
                deviations.append(deviation)

            feature_val = max(deviations, key=lambda x: abs(x))
            seg2_features.append(feature_val)

        normal_data["seg2_feature"] = seg2_features

        # 段4: P17-P20 端点差值法
        normal_data["seg4_feature"] = normal_data["P20"] - normal_data["P17"]

        # 第三段物理分类
        classifications = []
        feature_details = []

        for idx, row in normal_data.iterrows():
            try:
                # 获取P9-P16的数值
                p_values = np.array([row[f"P{i}"] for i in range(9, 17)])
                # 获取所有点位的平均值
                avg_all_points = np.mean([row[f"P{i}"] for i in range(1, 21)])
                # 特征1：整体趋势 (P9到P17的斜率) （舍弃）
                # trend = np.mean([row[f'P{i}'] for i in range(9, 18)])  # P9-P17的均值

                # 特征2：段内标准差 (P9-P17的离散程度)（舍弃）
                # std_dev = np.std(p_values)

                # 特征3：前后半段斜率差 (WAVE专用)（舍弃）
                # slope_left = (row['P13'] - row['P9']) / 4  # P9-P13斜率
                # slope_right = (row['P17'] - row['P13']) / 4  # P13-P17斜率
                # slope_diff = abs(slope_right - slope_left)
                # start = row['P9']
                # 用二次函数拟合判断凸凹
                x = np.arange(9)  # [0,1,2,...,8]
                y = [row[f"P{i}"] for i in range(9, 18)]  # P9-P17的值
                a = np.polyfit(x, y, 2)[0]  # 二次项系数（曲率）
                # 判断凹凸

                upOrDown = calculateUpOrDown(p_values, avg_all_points)
                # 波峰波谷
                peaks, _ = find_peaks(p_values, height=0.05)
                valleys, _ = find_peaks(-p_values, height=0.05)

                # 特征：振幅（辅助）
                amplitude = np.max(p_values) - np.min(p_values)
                # 分类逻辑
                if len(peaks) >= 1 and len(valleys) >= 1 and amplitude >= 0.08:
                    category = "WAVE"
                elif upOrDown == 1:
                    category = "ARC_UP"
                elif upOrDown == 2:
                    category = "ARC_DOWN"
                elif amplitude <= 0.065:
                    category = "FLAT"
                elif a < -0.0008:  # 曲率为负 → 开口向下 → 上凸（ARC_UP）
                    category = "ARC_UP"
                elif a > 0.0008:  # 曲率为正 → 开口向上 → 下凹（ARC_DOWN）
                    category = "ARC_DOWN"
                else:
                    category = "FLAT"  # 接近直线

                # # 规则2：ARC（圆弧型）- 只要有明确趋势
                # elif trend > start + 0.015:  # 上凸趋势明显
                #     category = 'ARC_UP'
                # elif trend < start - 0.015:  # 下凹趋势明显
                #     category = 'ARC_DOWN'

                # # 规则3：FLAT（平缓型）- 其他全部归为此类
                # # 条件：波动小 或 趋势弱
                # else:
                #     # 兜底判断：如果连FLAT都不算严格，就归为FLAT_LIGHT（轻微变形）
                #     if std_dev < 0.04:
                #         category = 'FLAT'
                #     else:
                #         category = 'FLAT'  # 新增：轻微起伏但无明确方向

                classifications.append(category)
                feature_details.append(
                    {
                        "amplitude": amplitude,
                        "peaks_count": len(peaks),
                        "valleys_count": len(valleys),
                        "curvature": a,
                        "up_or_down": upOrDown,
                    }
                )

            except Exception as e:
                logger.warning(f"计算第{idx}行第三段分类时出错: {e}")
                classifications.append("UNKNOWN")
                feature_details.append(
                    {
                        "amplitude": np.nan,
                        "peaks_count": 0,
                        "valleys_count": 0,
                        "curvature": np.nan,
                        "up_or_down": 0,
                    }
                )

        normal_data["seg3_category"] = classifications

        # 添加特征详细信息
        for key in [
            "amplitude",
            "peaks_count",
            "valleys_count",
            "curvature",
            "up_or_down",
        ]:
            values = [details[key] for details in feature_details]
            normal_data[f"seg3_{key}"] = values

        # 进行P/N分类
        thresholds = [0, 0, 0]  # 对应段1, 2, 4

        # 段1分类
        normal_data["seg1_label"] = normal_data["seg1_feature"].apply(
            lambda x: "P" if x >= thresholds[0] else "N"
        )

        # 段2分类
        normal_data["seg2_label"] = normal_data["seg2_feature"].apply(
            lambda x: "P" if x >= thresholds[1] else "N"
        )

        # 段4分类
        normal_data["seg4_label"] = normal_data["seg4_feature"].apply(
            lambda x: "P" if x >= thresholds[2] else "N"
        )

        # 段3物理标签映射
        physical_to_label = {
            "ARC_UP": "A",  # 上圆弧 -> A
            "ARC_DOWN": "R",  # 下圆弧 -> R (代表凹)
            "FLAT": "F",  # 平缓型 -> F
            "WAVE": "W",  # 剧烈波动 -> W
            "UNKNOWN": "U",
        }

        normal_data["seg3_label"] = normal_data["seg3_category"].map(physical_to_label)

        # 组合4段Shape
        normal_data["shape_v4"] = (
            normal_data["seg1_label"]
            + normal_data["seg2_label"]
            + normal_data["seg3_label"]
            + normal_data["seg4_label"]
        )

        # 统计结果
        print("V4版本物理分类结果:")
        print("=" * 60)
        print(f"总处理数据: {len(normal_data)} 条")

        # 物理分类分布
        seg3_counts = normal_data["seg3_category"].value_counts()
        print("\\n第三段物理分类分布:")
        for category, count in seg3_counts.items():
            percentage = count / len(normal_data) * 100
            print(f"  {category}: {count} 条 ({percentage:.1f}%)")

        # 物理分类特征统计
        print("\\n物理分类特征统计:")
        for category in ["ARC_UP", "ARC_DOWN", "FLAT", "WAVE"]:
            if category in seg3_counts:
                group = normal_data[normal_data["seg3_category"] == category]
                avg_amplitude = group["seg3_amplitude"].mean()
                avg_curvature = group["seg3_curvature"].mean()
                avg_up_or_down = group["seg3_up_or_down"].mean()
                print(
                    f"  {category}: 振幅={avg_amplitude:.3f}, 曲率={avg_curvature:.5f}, 凸凹={avg_up_or_down:.2f}"
                )

        # Shape分布
        shape_counts = normal_data["shape_v4"].value_counts()
        print(f"\\nShape分布 (前10个):")
        for shape, count in shape_counts.head(10).items():
            percentage = count / len(normal_data) * 100
            print(f"  {shape}: {count} 条 ({percentage:.1f}%)")

        # 保存结果
        normal_data.to_csv("test_v4_results.csv", index=False)
        print(f"\\n结果已保存到 test_v4_results.csv")

        return normal_data

    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback

        traceback.print_exc()
        return None


def calculateUpOrDown(p_values, avg_all_points):
    if len(p_values) < 3:
        return False

    max_index = np.argmax(p_values)
    min_index = np.argmin(p_values)
    judge = np.max(p_values) - avg_all_points > avg_all_points - np.min(p_values)
    if judge:
        # 使用 numpy 向量化操作
        left_increasing = np.sum(np.diff(p_values[: max_index + 1]) > 0)
        right_decreasing = np.sum(np.diff(p_values[max_index:]) < 0)

        left_ratio = left_increasing / max_index if max_index > 0 else 0
        right_ratio = (
            right_decreasing / (len(p_values) - max_index - 1)
            if max_index < len(p_values) - 1
            else 0
        )
        if left_ratio > 0.6 and right_ratio > 0.6:
            return 1
        return 0
    else:
        left_decreasing = np.sum(np.diff(p_values[: min_index + 1]) < 0)  # 左边下降
        right_increasing = np.sum(np.diff(p_values[min_index:]) > 0)  # 右边上升

        left_ratio = left_decreasing / min_index if min_index > 0 else 0
        right_ratio = (
            right_increasing / (len(p_values) - min_index - 1)
            if min_index < len(p_values) - 1
            else 0
        )
        if left_ratio > 0.6 and right_ratio > 0.6:
            return 2
        return 0
    return 0


if __name__ == "__main__":
    result = test_v4_physical_classification()
