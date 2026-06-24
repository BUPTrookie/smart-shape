"""
影响系数可视化 - V4版本（每个X坐标的拟合曲线）
================================================

为每个X坐标单独绘制:
1. 距离-影响散点图
2. 线性拟合曲线
3. 相关系数和R²值
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# 文件路径
JSON_PATH = "Output/RS_impact_analysis/influence_coefficients.json"
OUTPUT_DIR = Path("Output/influence_visualization_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 点位位置（从P1到P20的实际X坐标）
POINT_POSITIONS = {
    'P1': -70.99, 'P2': -68.98, 'P3': -65.66, 'P4': -62.12, 'P5': -59.46,
    'P6': -55.33, 'P7': -51.19, 'P8': -47.06, 'P9': -42.93, 'P10': -32.25,
    'P11': -27.20, 'P12': -21.60, 'P13': -15.99, 'P14': -10.38, 'P15': -4.77,
    'P16': 0.84, 'P17': 6.44, 'P18': 12.05, 'P19': 17.66, 'P20': 22.28
}


def load_influence_data():
    """加载影响系数JSON"""
    logger.info(f"加载数据: {JSON_PATH}")

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    influence = data['influence_coefficients']

    return influence


def get_x_coord_data(influence, x_coord):
    """获取指定X坐标的影响系数数据"""
    data = {
        'left': {},   # RS1/RS3 (负值X坐标)
        'right': {}   # RS2/RS4 (正值X坐标)
    }

    for rs_name in ['RS1', 'RS2', 'RS3', 'RS4']:
        if rs_name not in influence:
            continue

        # 判断是左侧还是右侧 (按X坐标符号分组)
        position = 'left' if rs_name in ['RS1', 'RS3'] else 'right'
        pos_str = str(x_coord)

        if pos_str not in influence[rs_name]:
            continue

        # 获取该X坐标下所有点位的影响系数
        point_data = influence[rs_name][pos_str]

        for point_name, coef in point_data.items():
            if point_name not in data[position]:
                data[position][point_name] = []

            data[position][point_name].append({
                'rs_name': rs_name,
                'coefficient': coef
            })

    return data


def calculate_distance_decay(x_coord, position_data):
    """计算距离衰减数据"""
    point_info = []

    for point_name in [f'P{i}' for i in range(1, 21)]:
        target_point = f'delta_{point_name}'
        point_pos = POINT_POSITIONS[point_name]

        # 计算距离
        distance = abs(point_pos - x_coord)

        # 获取影响系数（平均）
        if target_point in position_data and len(position_data[target_point]) > 0:
            coeffs = [abs(item['coefficient']) for item in position_data[target_point]]
            effect = np.mean(coeffs)
            n_samples = len(coeffs)
        else:
            effect = 0
            n_samples = 0

        if effect > 0:
            point_info.append({
                'point': point_name,
                'point_pos': point_pos,
                'distance': distance,
                'effect': effect,
                'n_samples': n_samples
            })

    return point_info


def plot_single_x_coord(x_coord, point_data_upper, point_data_lower, output_path):
    """为单个X坐标绘制影响系数分布图"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 左侧压头
    ax1 = axes[0]

    if len(point_data_upper) > 1:
        # X轴: 点位索引 (1-20)
        point_indices = [int(p['point'][1:]) for p in point_data_upper]  # P1->1, P2->2, ...
        effects = [p['effect'] for p in point_data_upper]
        points = [p['point'] for p in point_data_upper]

        # 散点图
        ax1.scatter(point_indices, effects, alpha=0.6, s=80, c='steelblue', edgecolors='black', linewidth=0.5)

        # 标注点位名称
        for i, point in enumerate(points):
            ax1.annotate(point, (point_indices[i], effects[i]),
                        fontsize=8, alpha=0.7,
                        xytext=(0, 5), textcoords='offset points', ha='center')

        # 线性拟合 (仍然用距离计算相关性,但图上显示点位)
        distances = [p['distance'] for p in point_data_upper]
        slope, intercept, r_value, p_value, std_err = stats.linregress(distances, effects)
        r_squared = r_value ** 2

        # 绘制拟合线 (在点位空间上)
        # 将距离转换回点位坐标来绘制趋势线
        min_dist_idx = point_indices[np.argmin(distances)]
        max_dist_idx = point_indices[np.argmax(distances)]
        min_dist_val = min(distances)
        max_dist_val = max(distances)

        # 在两个点位之间画拟合线
        line_y = [slope * min_dist_val + intercept, slope * max_dist_val + intercept]
        ax1.plot([min_dist_idx, max_dist_idx], line_y, 'r-', linewidth=2, alpha=0.7, label=f'Fit: slope={slope:.4f}')

        # 添加统计信息
        stats_text = f'Corr: {r_value:.3f}\nR²: {r_squared:.3f}\nP: {p_value:.2e}\nN: {len(point_data_upper)}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax1.set_xlabel('Measurement Point (P1-P20)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Influence Coefficient (abs)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Press Position X = {x_coord:.2f} mm - Left (RS1/RS3)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(1, 21))
        ax1.set_xticklabels([f'P{i}' for i in range(1, 21)], rotation=45, ha='right', fontsize=8)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 标注压头施加位置 - 找到最接近压头X坐标的点位
        closest_point = min(POINT_POSITIONS.keys(), key=lambda p: abs(POINT_POSITIONS[p] - x_coord))
        closest_idx = int(closest_point[1:])  # P10 -> 10

        # 绘制垂直虚线标注压头位置
        ax1.axvline(x=closest_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(closest_idx, ax1.get_ylim()[1] * 0.95, f'Press Here\nX={x_coord:.2f}',
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                color='red')

        # 判断衰减规律
        if slope < -0.001:
            decay_status = 'Distance Decay (Negative) ✓'
            color_status = 'green'
        elif slope > 0.001:
            decay_status = 'Distance Increase (Positive) ✗'
            color_status = 'red'
        else:
            decay_status = 'No Correlation'
            color_status = 'gray'

        ax1.text(0.5, 0.05, decay_status, transform=ax1.transAxes,
                fontsize=11, fontweight='bold', color=color_status,
                ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    else:
        ax1.text(0.5, 0.5, f'Insufficient Data\n(n={len(point_data_upper)})',
                transform=ax1.transAxes, fontsize=14, ha='center', va='center')
        ax1.set_title(f'Press Position X = {x_coord:.2f} mm - Left (RS1/RS3)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(1, 21))
        ax1.set_xticklabels([f'P{i}' for i in range(1, 21)], rotation=45, ha='right', fontsize=8)

    # 右图: 右侧压头
    ax2 = axes[1]

    if len(point_data_lower) > 1:
        # X轴: 点位索引 (1-20)
        point_indices = [int(p['point'][1:]) for p in point_data_lower]
        effects = [p['effect'] for p in point_data_lower]
        points = [p['point'] for p in point_data_lower]

        # 散点图
        ax2.scatter(point_indices, effects, alpha=0.6, s=80, c='darkorange', edgecolors='black', linewidth=0.5)

        # 标注点位名称
        for i, point in enumerate(points):
            ax2.annotate(point, (point_indices[i], effects[i]),
                        fontsize=8, alpha=0.7,
                        xytext=(0, 5), textcoords='offset points', ha='center')

        # 线性拟合
        distances = [p['distance'] for p in point_data_lower]
        slope, intercept, r_value, p_value, std_err = stats.linregress(distances, effects)
        r_squared = r_value ** 2

        # 绘制拟合线
        min_dist_idx = point_indices[np.argmin(distances)]
        max_dist_idx = point_indices[np.argmax(distances)]
        min_dist_val = min(distances)
        max_dist_val = max(distances)

        line_y = [slope * min_dist_val + intercept, slope * max_dist_val + intercept]
        ax2.plot([min_dist_idx, max_dist_idx], line_y, 'r-', linewidth=2, alpha=0.7, label=f'Fit: slope={slope:.4f}')

        # 添加统计信息
        stats_text = f'Corr: {r_value:.3f}\nR²: {r_squared:.3f}\nP: {p_value:.2e}\nN: {len(point_data_lower)}'
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_xlabel('Measurement Point (P1-P20)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Influence Coefficient (abs)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Press Position X = {x_coord:.2f} mm - Right (RS2/RS4)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(1, 21))
        ax2.set_xticklabels([f'P{i}' for i in range(1, 21)], rotation=45, ha='right', fontsize=8)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 标注压头施加位置
        closest_point = min(POINT_POSITIONS.keys(), key=lambda p: abs(POINT_POSITIONS[p] - x_coord))
        closest_idx = int(closest_point[1:])

        ax2.axvline(x=closest_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.text(closest_idx, ax2.get_ylim()[1] * 0.95, f'Press Here\nX={x_coord:.2f}',
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                color='red')

        # 判断衰减规律
        if slope < -0.001:
            decay_status = 'Distance Decay (Negative) ✓'
            color_status = 'green'
        elif slope > 0.001:
            decay_status = 'Distance Increase (Positive) ✗'
            color_status = 'red'
        else:
            decay_status = 'No Correlation'
            color_status = 'gray'

        ax2.text(0.5, 0.05, decay_status, transform=ax2.transAxes,
                fontsize=11, fontweight='bold', color=color_status,
                ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    else:
        ax2.text(0.5, 0.5, f'Insufficient Data\n(n={len(point_data_lower)})',
                transform=ax2.transAxes, fontsize=14, ha='center', va='center')
        ax2.set_title(f'Press Position X = {x_coord:.2f} mm - Right (RS2/RS4)', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(1, 21))
        ax2.set_xticklabels([f'P{i}' for i in range(1, 21)], rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  保存: {output_path.name}")


def create_summary_table(all_results, output_path):
    """创建汇总表"""
    rows = []

    for result in all_results:
        row = {
            'X_coord_mm': f"{result['x_coord']:.2f}",
            'Upper_corr': f"{result['upper_corr']:.3f}" if result['upper_corr'] is not None else 'N/A',
            'Upper_slope': f"{result['upper_slope']:.4f}" if result['upper_slope'] is not None else 'N/A',
            'Upper_R2': f"{result['upper_R2']:.3f}" if result['upper_R2'] is not None else 'N/A',
            'Upper_N': result['upper_N'],
            'Lower_corr': f"{result['lower_corr']:.3f}" if result['lower_corr'] is not None else 'N/A',
            'Lower_slope': f"{result['lower_slope']:.4f}" if result['lower_slope'] is not None else 'N/A',
            'Lower_R2': f"{result['lower_R2']:.3f}" if result['lower_R2'] is not None else 'N/A',
            'Lower_N': result['lower_N']
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n保存汇总表: {output_path.name}")
    logger.info(f"\n{df.to_string(index=False)}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("影响系数可视化 - V4版本（每个X坐标的拟合曲线）")
    logger.info("=" * 80)

    # 1. 加载数据
    influence = load_influence_data()

    # 2. 获取所有X坐标
    x_coords = set()
    for rs_name in ['RS1', 'RS2', 'RS3', 'RS4']:
        if rs_name not in influence:
            continue
        for pos_str in influence[rs_name].keys():
            x_coords.add(float(pos_str))

    x_coords = sorted(x_coords)
    logger.info(f"\n发现 {len(x_coords)} 个不同的X坐标:")
    logger.info(f"  {x_coords}")

    # 3. 为每个X坐标绘制拟合曲线
    logger.info(f"\n生成各X坐标的拟合曲线...")
    all_results = []

    for x_coord in x_coords:
        logger.info(f"\n处理 X坐标 = {x_coord:.2f} mm:")

        # 获取数据
        coord_data = get_x_coord_data(influence, x_coord)

        # 计算距离衰减
        point_data_upper = calculate_distance_decay(x_coord, coord_data['left'])
        point_data_lower = calculate_distance_decay(x_coord, coord_data['right'])

        # 提取统计信息
        if len(point_data_upper) > 1:
            distances_upper = [p['distance'] for p in point_data_upper]
            effects_upper = [p['effect'] for p in point_data_upper]
            slope_upper, intercept_upper, r_value_upper, p_value_upper, std_err_upper = stats.linregress(distances_upper, effects_upper)
            upper_corr = r_value_upper
            upper_slope = slope_upper
            upper_R2 = r_value_upper ** 2
        else:
            upper_corr = None
            upper_slope = None
            upper_R2 = None

        if len(point_data_lower) > 1:
            distances_lower = [p['distance'] for p in point_data_lower]
            effects_lower = [p['effect'] for p in point_data_lower]
            slope_lower, intercept_lower, r_value_lower, p_value_lower, std_err_lower = stats.linregress(distances_lower, effects_lower)
            lower_corr = r_value_lower
            lower_slope = slope_lower
            lower_R2 = r_value_lower ** 2
        else:
            lower_corr = None
            lower_slope = None
            lower_R2 = None

        # 保存结果
        all_results.append({
            'x_coord': x_coord,
            'upper_corr': upper_corr,
            'upper_slope': upper_slope,
            'upper_R2': upper_R2,
            'upper_N': len(point_data_upper),
            'lower_corr': lower_corr,
            'lower_slope': lower_slope,
            'lower_R2': lower_R2,
            'lower_N': len(point_data_lower)
        })

        # 绘图
        plot_single_x_coord(
            x_coord,
            point_data_upper,
            point_data_lower,
            OUTPUT_DIR / f"distance_decay_x_{x_coord:.2f}.png"
        )

    # 4. 创建汇总表
    logger.info(f"\n生成汇总表...")
    create_summary_table(all_results, OUTPUT_DIR / "distance_decay_summary.csv")

    # 5. 生成总结报告
    logger.info(f"\n生成总结报告...")
    with open(OUTPUT_DIR / "analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Distance Decay Analysis - By X Coordinate\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total X coordinates: {len(x_coords)}\n")
        f.write(f"{x_coords}\n\n")

        f.write("=" * 80 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            f.write(f"X = {result['x_coord']:.2f} mm:\n")

            if result['upper_corr'] is not None:
                status_upper = "NEGATIVE (OK)" if result['upper_slope'] < 0 else "POSITIVE (UNEXPECTED)" if result['upper_slope'] > 0 else "NEUTRAL"
                f.write(f"  Upper: corr={result['upper_corr']:.3f}, slope={result['upper_slope']:.4f}, R²={result['upper_R2']:.3f}, N={result['upper_N']} [{status_upper}]\n")
            else:
                f.write(f"  Upper: Insufficient data (N={result['upper_N']})\n")

            if result['lower_corr'] is not None:
                status_lower = "NEGATIVE (OK)" if result['lower_slope'] < 0 else "POSITIVE (UNEXPECTED)" if result['lower_slope'] > 0 else "NEUTRAL"
                f.write(f"  Lower: corr={result['lower_corr']:.3f}, slope={result['lower_slope']:.4f}, R²={result['lower_R2']:.3f}, N={result['lower_N']} [{status_lower}]\n")
            else:
                f.write(f"  Lower: Insufficient data (N={result['lower_N']})\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Key Findings\n")
        f.write("=" * 80 + "\n\n")

        # 统计符合预期的
        ok_upper = sum(1 for r in all_results if r['upper_slope'] is not None and r['upper_slope'] < 0)
        total_upper = sum(1 for r in all_results if r['upper_slope'] is not None)

        ok_lower = sum(1 for r in all_results if r['lower_slope'] is not None and r['lower_slope'] < 0)
        total_lower = sum(1 for r in all_results if r['lower_slope'] is not None)

        f.write(f"Upper press: {ok_upper}/{total_upper} coordinates show negative decay\n")
        f.write(f"Lower press: {ok_lower}/{total_lower} coordinates show negative decay\n\n")

        f.write("Output files:\n")
        f.write("  - distance_decay_x_*.png: Individual plots for each X coordinate\n")
        f.write("  - distance_decay_summary.csv: Statistical summary table\n")
        f.write("  - analysis_report.txt: This report\n")

    logger.info(f"\n" + "=" * 80)
    logger.info(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    logger.info(f"=" * 80)


if __name__ == "__main__":
    main()
