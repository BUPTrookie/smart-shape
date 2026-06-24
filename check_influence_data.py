"""
检查影响系数JSON的数据合理性
"""
import json

json_path = "Output/RS_impact_analysis/influence_coefficients.json"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

influence = data['influence_coefficients']
feature_names = data['metadata']['feature_names']

print("=" * 80)
print("特征名检查 (位置特征)")
print("=" * 80)

# 找出所有位置特征
pos_features = [fn for fn in feature_names if '_pos_' in fn]

for rs_name in ['RS1', 'RS2', 'RS3', 'RS4']:
    rs_features = [fn for fn in pos_features if fn.startswith(rs_name)]
    print(f"\n{rs_name} 位置特征 ({len(rs_features)} 个):")

    # 提取位置值
    positions = []
    for feat in rs_features:
        pos_str = feat.split('_pos_')[1]
        pos_val = float(pos_str)
        positions.append(pos_val)

    positions_sorted = sorted(positions)
    print(f"  位置范围: {positions_sorted[0]:.2f} ~ {positions_sorted[-1]:.2f}")
    print(f"  所有位置: {positions_sorted}")

print("\n" + "=" * 80)
print("建议:")
print("  1. 检查 rs_impact_config.py 中的 RS_COLUMNS 定义")
print("  2. 检查原始数据中 RSX 的取值分布")
print("  3. 确认 RSX=-70.0 是否合理")
