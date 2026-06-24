"""
检查原始数据中RSX的取值分布,找出异常值来源
"""
import pandas as pd
import json

print("=" * 80)
print("原始数据 RSX 分布检查")
print("=" * 80)

# 1. 加载原始数据
df = pd.read_excel("Data/total_final_processed.xlsx", sheet_name="Reshaping")

print(f"\n总数据量: {len(df)} 条")

# 2. 检查每个RSX的取值分布
for rs_name in ['RS1', 'RS2', 'RS3', 'RS4']:
    rsx_col = f'{rs_name}X'
    rsz_col = f'{rs_name}Z'

    print(f"\n{'='*80}")
    print(f"{rs_name} 位置 (X) 和下压量 (Z) 分析:")
    print(f"{'='*80}")

    if rsx_col not in df.columns:
        print(f"  列 {rsx_col} 不存在!")
        continue

    # 过滤非空数据
    valid_data = df[df[rsx_col].notna()][[rsx_col, rsz_col]]
    print(f"  有效数据量: {len(valid_data)} 条")

    if len(valid_data) == 0:
        print(f"  无有效数据!")
        continue

    # X位置分布
    x_values = valid_data[rsx_col].value_counts().sort_index()
    print(f"\n  {rsx_col} 取值分布 (共 {len(x_values)} 个不同值):")
    print(f"  {'位置值':>10} {'出现次数':>10} {'占比(%)':>10}")
    print(f"  {'-'*35}")
    for pos, count in x_values.items():
        pct = count / len(valid_data) * 100
        print(f"  {pos:>10.2f} {count:>10} {pct:>9.2f}%")

    print(f"\n  位置范围: {x_values.index.min():.2f} ~ {x_values.index.max():.2f}")

    # Z下压量分布
    z_values = valid_data[rsz_col]
    print(f"\n  {rsz_col} 下压量统计:")
    print(f"    最小值: {z_values.min():.2f}")
    print(f"    最大值: {z_values.max():.2f}")
    print(f"    平均值: {z_values.mean():.2f}")
    print(f"    中位数: {z_values.median():.2f}")

    # 检查异常位置值
    if -70.0 in x_values.index:
        print(f"\n  [WARNING] 发现异常位置值: -70.0 (出现 {x_values[-70.0]} 次)")
        # 显示这些样本的条码
        samples_with_neg70 = df[df[rsx_col] == -70.0][['Barcode', rsx_col, rsz_col]].head(10)
        print(f"\n  包含 -70.0 的样本示例:")
        for idx, row in samples_with_neg70.iterrows():
            print(f"    条码: {row['Barcode']}, {rsx_col}={row[rsx_col]}, {rsz_col}={row[rsz_col]}")

print("\n" + "=" * 80)
print("建议:")
print("  1. 检查数据采集过程，确认 -70.0 是否为错误值")
print("  2. 如果是错误值，需要在数据预处理阶段进行修正")
print("  3. 检查 rs_impact_config.py 中 POINT_X_COORDS 的定义是否与实际压头位置混淆")
print("=" * 80)
