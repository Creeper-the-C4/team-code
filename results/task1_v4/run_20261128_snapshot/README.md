# 第一题 V4 早期运行快照

快照时间：**2026年11月28日 10:57:10（北京时间）**。记录时第一题8组配置、六折训练正在进行，当时已完成 **2/48折**、**0/8组完整配置**。当前 `F1H_2s`。现已完成全组实验，见 [完成结果报告](../../../docs/09_task1_v4_full_validation_report.md)。

实验源代码提交：`46bc8a1056461a3980a75d51869c4a676d5dbfcc`。今日任务与验收要求见 [实验安排](../../../docs/08_2026-11-28_experiment_plan_and_progress.md)。

| 文件 | 内容 |
|---|---|
| [run_manifest.json](run_manifest.json) | 数据划分、窗口索引与源代码哈希、配置、环境及实际执行脚本哈希 |
| [progress_snapshot.json](progress_snapshot.json) | 读取时的云端状态和已完成折次；属于带时间快照 |
| [feature_cache_audit.json](feature_cache_audit.json) | 300份缓存按特征与窗口汇总的缺失统计、F3旧基线抽样比对与步频缺失复核 |
| [first_fold_model_audit.json](first_fold_model_audit.json) | 首折测试索引、标签和保存模型的512窗口重载预测核验 |
| [执行入口](execution_scripts/run_task1_v4_cloud.py) | 实际使用的第一题独立运行脚本，按manifest哈希原样保存 |
| [模型保存包装](execution_scripts/task1_v4_train_with_models.py) | 调用上游训练流程并额外保存已拟合模型；不改变训练或预测逻辑 |

这些脚本为本轮AutoDL路径与固定版本的执行记录，复制到其他路径前需调整相应配置；作为证据副本保存，不替代仓库原有训练入口。原始数据、大特征数组、模型与逐窗口预测保留云端。

辅助校验曾在训练开始前误判正常的步频缺失，已修正并归档。修正后300份缓存核验通过；初次中断记录保存在云端本轮结果目录的 `archive/`。

云端结果目录：`/root/autodl-tmp/C_Problem/results/task1_v4_20261128`；实时进度读取该目录的 `stage_status.json`。本地快照不自动刷新。

公开发布时将逐人缓存明细和人员列表改为数量与摘要，原始详单保留云端与本地工作区；算法执行脚本仍按原哈希保存。公开元数据副本不用于恢复运行。
