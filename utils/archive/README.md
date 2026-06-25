# 归档：早期多 sheet xlsx 预处理工具

本目录存放项目早期（数据源为多 sheet 的 `Data/total.xlsx` 时期）的一次性数据预处理脚本：

- `process_all_sheets_final.py`
- `process_excel_sheets_clean.py`
- `process_multi_sheets_excel.py`
- `process_pre_data.py`
- `process_pre_excel.py`

这些脚本是同一功能（多 sheet Pre 数据抽取/字段转移）的多次迭代版本，相互重复。
数据源已切换为单文件 `Data/total.csv`（见 `rs_impact_config.INPUT_DATA_PATH`），
当前流程（`rs_impact_analyzer` 配对、`db.import_history` 导入）**不再使用**这些脚本，
核心代码也**不引用** `utils/`。

故从工作区归档至此、保留 git 历史，**不合并**（合并出处理废弃数据源的脚本属负价值）。
如需回顾早期多 sheet 处理逻辑，可在此查阅。
