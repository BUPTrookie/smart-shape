import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress


def calculate_up_or_down(p_values, avg_all_points):
    """
    计算凸凹判断的复合函数（与test_v4_simple.py保持一致）
    返回值：1-上凸(ARC_UP)，2-下凹(ARC_DOWN)，0-不明确
    """
    if len(p_values) < 3:
        return 0

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


def segment1_endpoint_diff(p1, p4):
    """
    段1：端点差值法 P4-P1
    阈值：0，分类为P/N
    """
    diff = p4 - p1
    return "P" if diff >= 0 else "N", diff


def segment2_linearity_fitting(p5, p6, p7, p8):
    """
    段2：端点法直线度拟合
    计算线性度误差
    阈值：0，分类为P/N
    """
    p_values = [p5, p6, p7, p8]
    start_val = p5
    end_val = p8
    endpoint_diff = end_val - start_val

    deviations = []
    for i in range(4):
        actual_val = p_values[i]
        theoretical_val = start_val + i * (endpoint_diff / 3)
        deviation = actual_val - theoretical_val
        deviations.append(deviation)

    # 计算最大偏差（保留符号信息，与test_v4_simple.py一致）
    feature_val = max(deviations, key=lambda x: abs(x))

    # 阈值：0，分类为P/N（基于偏差值本身，不是绝对值）
    return "P" if feature_val >= 0 else "N", feature_val, deviations


def segment3_feature_classification(all_points, p9_to_p16):
    """
    段3：特征值计算（参考第三段分类流程）
    """
    # 转换为numpy数组
    p9_to_p16 = np.array(p9_to_p16)
    all_points = np.array(all_points)

    # 计算振幅
    amplitude = np.max(p9_to_p16) - np.min(p9_to_p16)

    # 计算全局均值
    avg_all_points = np.mean(all_points)

    # 寻找峰谷
    peaks, _ = find_peaks(p9_to_p16, height=0.05)
    valleys, _ = find_peaks(-p9_to_p16, height=0.05)

    # 计算曲率
    x = np.arange(len(p9_to_p16))
    coeffs = np.polyfit(x, p9_to_p16, 2)
    a = coeffs[0]  # 二次项系数

    # 凸凹判断
    upOrDown = calculate_up_or_down(p9_to_p16, avg_all_points)

    # 分类逻辑
    if len(peaks) >= 1 and len(valleys) >= 1 and amplitude >= 0.08:
        category = "W"
    elif upOrDown == 1:
        category = "A"
    elif upOrDown == 2:
        category = "R"
    elif amplitude <= 0.065:
        category = "F"
    elif a < -0.0008:  # 曲率为负 → 开口向下 → 上凸（ARC_UP）
        category = "A"
    elif a > 0.0008:  # 曲率为正 → 开口向上 → 下凹（ARC_DOWN）
        category = "R"
    else:
        category = "F"  # 接近直线

    return category, {
        "amplitude": amplitude,
        "peaks_count": len(peaks),
        "valleys_count": len(valleys),
        "curvature": a,
        "up_or_down": upOrDown,
    }


def segment4_endpoint_diff(p17, p20):
    """
    段4：端点差值法 P20-P1
    阈值：0，分类为P/N
    """
    diff = p20 - p17
    return "P" if diff >= 0 else "N", diff


def process_data(input_file, output_file):
    """
    处理数据的主函数
    """
    # 读取数据
    df = pd.read_csv(input_file)

    # 准备结果列表
    results = []

    for idx, row in df.iterrows():
        # 提取P1-P20的数据
        p_columns = [f"FAI156-P{i}" for i in range(1, 21)]
        p_values = [row[col] for col in p_columns]

        # 段1：P1-P4 (索引0-3)
        shape1, diff1 = segment1_endpoint_diff(p_values[0], p_values[3])

        # 段2：P5-P8 (索引4-7)
        shape2, error2, deviations2 = segment2_linearity_fitting(
            p_values[4], p_values[5], p_values[6], p_values[7]
        )

        # 段3：P9-P16 (索引8-15)
        p9_to_p16 = p_values[8:16]
        shape3, features3 = segment3_feature_classification(p_values, p9_to_p16)

        # 段4：P1-P20 (索引0-19)
        shape4, diff4 = segment4_endpoint_diff(p_values[16], p_values[19])

        # 创建结果行
        result_row = row.to_dict()  # 转换为字典避免数据类型问题
        result_row["Shape"] = f"{shape1}{shape2}{shape3}{shape4}"

        # 添加详细特征
        result_row["Seg1_Shape"] = shape1
        result_row["Seg1_Diff"] = diff1
        result_row["Seg2_Shape"] = shape2
        result_row["Seg2_Error"] = error2
        result_row["Seg2_Deviations"] = str(deviations2)  # 将偏差列表转为字符串保存
        result_row["Seg3_Shape"] = shape3
        result_row["Seg3_Amplitude"] = features3["amplitude"]
        result_row["Seg3_Peaks"] = features3["peaks_count"]
        result_row["Seg3_Valleys"] = features3["valleys_count"]
        result_row["Seg3_Curvature"] = features3["curvature"]
        result_row["Seg3_UpOrDown"] = features3["up_or_down"]
        result_row["Seg4_Shape"] = shape4
        result_row["Seg4_Diff"] = diff4

        results.append(result_row)

    # 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 确保Shape列为字符串类型，避免FutureWarning
    result_df["Shape"] = result_df["Shape"].astype("str")

    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"处理完成，结果保存至: {output_file}")
    print(f"总共处理了 {len(result_df)} 条记录")

    # 打印统计信息
    print("\n分段统计信息:")
    print(
        f"段1 - P类: {sum(result_df['Seg1_Shape'] == 'P')}, N类: {sum(result_df['Seg1_Shape'] == 'N')}"
    )
    print(
        f"段2 - P类: {sum(result_df['Seg2_Shape'] == 'P')}, N类: {sum(result_df['Seg2_Shape'] == 'N')}"
    )
    print(
        f"段3 - W: {sum(result_df['Seg3_Shape'] == 'W')}, A: {sum(result_df['Seg3_Shape'] == 'A')}, R: {sum(result_df['Seg3_Shape'] == 'R')}, F: {sum(result_df['Seg3_Shape'] == 'F')}"
    )
    print(
        f"段4 - P类: {sum(result_df['Seg4_Shape'] == 'P')}, N类: {sum(result_df['Seg4_Shape'] == 'N')}"
    )


if __name__ == "__main__":
    input_file = "Data/X9600DZ/data.csv"
    output_file = "test_1209_DZ_result.csv"

    process_data(input_file, output_file)
