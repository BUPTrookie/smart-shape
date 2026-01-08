import pandas as pd
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_by_shape_v4():
    """根据shape_v4类型分组,输出不同的CSV文件"""

    try:
        # 读取数据
        logger.info("正在读取 test_v4_results.csv ...")
        df = pd.read_csv("test_v4_results.csv")
        logger.info(f"成功读取 {len(df)} 条记录")

        # 获取所有shape_v4类型
        shape_types = df["shape_v4"].unique()
        logger.info(f"发现 {len(shape_types)} 种不同的shape_v4类型")

        # 创建输出目录
        output_dir = "Output/dz_group_by_shape_1126-1201"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建输出目录: {output_dir}")

        # 按shape_v4分组并保存
        group_stats = []
        for shape_type in sorted(shape_types):
            # 筛选该类型的数据
            group_df = df[df["shape_v4"] == shape_type]

            # 生成文件名
            filename = f"{output_dir}/shape_{shape_type}.csv"
            group_df.to_csv(filename, index=False, encoding="utf-8-sig")

            # 记录统计信息
            stats = {
                "shape_type": shape_type,
                "count": len(group_df),
                "percentage": len(group_df) / len(df) * 100,
            }
            group_stats.append(stats)

            logger.info(f"已保存: {filename} ({len(group_df)} 条记录)")

        # 输出统计汇总
        print("\n" + "=" * 60)
        print("分组统计汇总")
        print("=" * 60)
        print(f"总数据量: {len(df)} 条")
        print(f"分组数量: {len(shape_types)} 个")
        print("\n各分组详情:")

        # 按数量降序排列
        group_stats_sorted = sorted(group_stats, key=lambda x: x["count"], reverse=True)

        for stats in group_stats_sorted:
            print(
                f"  {stats['shape_type']}: {stats['count']} 条 ({stats['percentage']:.2f}%)"
            )

        print(f"\n所有文件已保存到目录: {output_dir}/")

        return group_stats_sorted

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    split_by_shape_v4()
