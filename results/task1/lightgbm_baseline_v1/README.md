# Task 1 LightGBM基线结果

版本：`task1_lightgbm_baseline_v1`；30人、220,702个2秒测试窗；完整六折已完成。

## 文件说明

| 文件 | 内容 |
|---|---|
| summary.json | 合并OOF、六折均值与标准差、聚类bootstrap区间及指标口径 |
| config.json | 固定参数、类别名、数据模式与脚本SHA256 |
| fold_metrics.csv | 每折分数、最佳迭代、验证分数与耗时 |
| per_class_oof.csv | 合并测试的逐类precision、recall、F1和支持数 |
| per_subject_metrics.csv | 30名测试参与者的指标，每人恰好测试一次 |
| confusion_oof.csv | 混淆矩阵计数，行为真值，列为预测 |
| confusion_pairs.csv | 按错误数量排序的类别混淆对 |
| window_group_metrics.csv | 全部、纯净、稳定、过渡和Others等分组表现 |
| per_class_*.csv | 各窗口分组的逐类指标，结合支持数读取 |
| confusion_oof_normalized.png | 按真实类别行归一化的混淆矩阵 |
| per_class_f1.png | 逐类F1图 |
| per_subject_accuracy.png | 跨参与者准确率 |
| Task1_LightGBM_baseline_report.txt | 中文解释报告 |
| report_bundle.zip | 上述小体积结果与本轮训练脚本的下载包 |

主指标为全部窗口上的13类Macro-F1：0.921536，准确率0.933467。12类指标同时提供“全部窗口含Others误报”和“只保留真实命名活动窗口”两种口径，不能混用。

分组或单人指标固定包含13类；若该组缺少某类，Macro-F1会受类别覆盖影响，须结合支持数和准确率阅读。

## 云端完整产物

```text
/root/autodl-tmp/C_Problem/results/task1_lightgbm_baseline_v1/
├── fold_0 至 fold_5/
│   ├── model.txt
│   ├── complete.json
│   ├── validation_history.json
│   └── test_predictions.parquet
└── oof_predictions.parquet
```

模型与逐窗概率数组保留云端，不进入Git历史。日志：`/root/autodl-tmp/C_Problem/logs/task1_lightgbm_baseline_v1.log`。

脚本：[scripts/task1](../../../scripts/task1/README.md)。数据底座：[independent_v1](../../preprocessing/independent_v1/README.md)。
