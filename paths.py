"""
统一路径管理模块
================

所有脚本应通过本模块获取项目路径，避免在各文件中硬编码
``Data/...`` ``Output/...`` 等相对路径（当前仍有 27 个脚本硬编码路径，
将逐步迁移到本模块，详见迁移说明）。

用法：
    from paths import PROJECT_ROOT, DATA_DIR, TOTAL_FINAL_PROCESSED

设计说明：
- PROJECT_ROOT 固定为 paths.py 所在目录（项目根），无论从哪个工作目录
  运行脚本都能正确定位，解决「换目录就崩」的问题。
- 新增脚本请优先使用本模块的常量；既有脚本的硬编码路径可在后续重构中
  逐步替换为本模块引用。
"""

from pathlib import Path

# 项目根目录（本文件所在目录）
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# 一级目录
DATA_DIR: Path = PROJECT_ROOT / "Data"
OUTPUT_DIR: Path = PROJECT_ROOT / "Output"
CHARTS_DIR: Path = PROJECT_ROOT / "charts"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
CONSTANTS_DIR: Path = PROJECT_ROOT / "constants"
UTILS_DIR: Path = PROJECT_ROOT / "utils"

# 常用数据文件
TOTAL_FINAL_PROCESSED: Path = DATA_DIR / "total_final_processed.xlsx"
RS_OUTPUT_DIR: Path = OUTPUT_DIR / "RS_impact_analysis"

# 测试数据目录（可选，用于存放小型测试夹具）
TESTS_DIR: Path = PROJECT_ROOT / "tests"


def ensure_output_dirs() -> None:
    """确保输出相关目录存在（运行分析/可视化前调用）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    RS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def as_str(path: Path) -> str:
    """将 Path 转为字符串（兼容需要 str 路径的旧接口，如 pandas.read_excel）。"""
    return str(path)
