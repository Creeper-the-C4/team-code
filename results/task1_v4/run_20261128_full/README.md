# 第一题 V4 全量实验汇总

**已完成8组配置、48折训练，结束时间为2026年11月28日 12:37:49（北京时间）。** 完整解释见 [结果报告](../../../docs/09_task1_v4_full_validation_report.md)，本目录也提供 [报告副本](completion_report.md)。

最高配置F1B、2秒、离线HMM的被试等权13类 Macro-F1为90.74%，合并窗口13类 Macro-F1为94.11%。这些是本轮固定六折上的探索性结果。

- [完整实验摘要](comparison_summary.json)：8组原始与HMM汇总指标。
- [完成状态](stage_status.json)：8组、48折及运行耗时，公开汇总副本。
- [运行记录](run_manifest.json)：源版本、配置、环境与脚本哈希；逐人索引摘要改为数量及总摘要。
- [特征核验汇总](feature_cache_audit.json)：300份缓存按特征与窗口汇总，逐人细节保留云端。
- [首折模型复核](first_fold_model_audit.json)：512窗口重载预测一致性。
- [发布清单](publication_manifest.json)：源报告包摘要、公开文件摘要和汇总转换说明。
- [压缩包清理记录](dataset_archive_deletion_receipt.json)：删除已解压dataset.zip及保留数据情况。
- [公开小报告包](report_bundle.zip)：本目录的公开版报告、指标和日志。

`task1_v4/` 中各配置包含 `metrics.csv`、`per_class_metrics.csv`、`fold_summary.csv`、`others_error_breakdown.csv`、步行三速混淆矩阵、每折HMM参数和完成记录。类别表含raw与hmm两种模式；三速3×3矩阵只显示三个步行类内部去向，三速F1仍由完整13类混淆统计计算。

`logs/` 保存8组训练日志。`execution_scripts/` 保存本次实际使用的独立运行入口、模型保存包装和只读进度查看程序；运行入口按AutoDL固定路径及源版本编写，作为复现记录保存。

仓库为公开状态。本次新结果不包含逐人指标、逐窗口预测、人体测量信息、原始信号及模型权重。JSON中的人员分配仅保留数量，相关详单继续保留云端，公开副本不用于断点恢复。

云端完整结果目录：`/root/autodl-tmp/C_Problem/results/task1_v4_20261128`。
