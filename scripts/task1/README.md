# Task 1基线测试脚本

- `task1_lightgbm_baseline.py`：固定LightGBM六折训练，仅验证参与者早停，测试参与者预测一次；保存模型、逐窗概率、指标及聚类bootstrap汇总。
- `task1_summarize.py`：读取已完成结果，生成图表与中文解释报告。不会重新训练或改变预测。

使用已有云端环境和独立数据底座：

```bash
PY=/root/autodl-tmp/C_Problem/.venv/bin/python
ROOT=/root/autodl-tmp/C_Problem/data/processed/independent_v1
OUT=/root/autodl-tmp/C_Problem/results/task1_lightgbm_baseline_v1

# 小规模检查仅使用训练与验证参与者
$PY scripts/task1/task1_lightgbm_baseline.py --root "$ROOT" --out /tmp/task1_smoke --smoke

# 正式六折 已完成且配置一致的折会跳过
$PY scripts/task1/task1_lightgbm_baseline.py --root "$ROOT" --out "$OUT"
$PY scripts/task1/task1_summarize.py --run "$OUT"
```

训练脚本与已运行版本SHA256一致，见结果 `config.json`。更改参数、脚本或特征定义后应使用新的输出目录，不覆盖本轮基线。

LightGBM使用12个CPU线程，最多300轮；固定学习率0.08、31叶、max_bin 63、min_child_samples 50、seed 42。早停只读取验证参与者的13类Macro-F1。

汇总脚本打包时从现有云端 `/root/autodl-tmp/C_Problem/scripts/task1_lightgbm_baseline.py` 读取原训练脚本；迁移其他环境需同步该路径或调整脚本根路径。环境依赖沿用 [预处理环境](../preprocessing/README.md)。
