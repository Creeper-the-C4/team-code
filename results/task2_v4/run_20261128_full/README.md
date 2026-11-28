# 第二问 V4 传感器精简全量结果

2026 年 11 月 28 日 22:04:59（北京时间）完成。四组各六折，共 2,856 次候选训练，30 人、220,702 个测试窗口。结论与固定佩戴原型的适用范围见 [完整报告](../../../docs/10_task2_v4_sensor_selection_report.md)。

| 文件 | 内容 |
|---|---|
| [数量曲线](all_curves.csv) | 四组各 17 个传感器数量的合并、被试等权与准确率 |
| [数量对照图](sensor_count_tradeoff.png) | 两种 Macro-F1 口径的曲线 |
| [类别汇总](class_metrics.csv) | 13 类 F1、支持数及相对全布局变化 |
| [分析与推荐原型](analysis_summary.json) | 配对区间、验证集数量规则、入选频率和耗时 |
| [运行与耗时](run_manifest.json) · [耗时表](timing_summary.csv) | 版本、数据切分与日志累计时间 |
| [第三问逐折布局](F1H_backward/per_fold_selected_subsets.json) | F1H 主实验各折各数量所选位置 |
| [复现脚本和 CUDA 修改](execution/) | 本轮实际运行入口、源码补丁和结果重算代码 |
| [导出校验表](source_export_checksums.json) | 云端导出文件的 SHA256 |

各组目录保留原始配置、结果汇总、逐折性能、位置入选频率、逐折布局和验证分数。报表按数量、动作或折次汇总。原始信号及逐窗口预测保留在云端。
