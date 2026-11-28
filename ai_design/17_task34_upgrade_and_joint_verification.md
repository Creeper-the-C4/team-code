# 第 10 轮多端协作归档：Task 3/4 精修、6D BiGRU 与联合复验并入 `main@d0532f2`

- **时间**：2026-11-28 16:57（北京时间）
- **基线与推进**：`origin/main` 从 `17bb054`（P0 30 人诊断 + Task 1 V4 48 折实测）推进至 **`d0532f2`**
- **执行策略**：严格**不中断云端正在进行中的 Task 2**；将本轮增量封装于独立接续脚本 `scripts/run_cloud_task3_task4_v4.sh`，待 Task 2 正常跑完并产出的 `per_fold_selected_subsets.json` 后无缝串联 Task 4 静态标定、Task 3 双版对比、BiGRU 6D 训练评估与 Task 1/2 论文出图。

---

## 一、提交链路与多端贡献

1. **VM-Claude (`1ed87b1` → `2329fa2`)**：
   - `T3-M3`（`src/task3/accel_factor.py`、`src/task3/smoother.py`、`src/task3/withheld_eval.py`、`scripts/task3/run_task3_wnoj.py`）：遵循 P0 `g` 裁决（乘 `ACCEL_G_TO_SI = 9.80665`，严禁重复去重力），**仅对存在已观测父节点和子节点的夹心留出节段启用局部鲁棒自由加速度因子**，含冲击掩蔽与 IRLS Huber 降权（$k=1.345$），解析雅可比有限差分校验精度优于 `2.2e-9`，并保留 `--accel-unit none`（`T3-M2`）逐比特隔离测试与主副对照报告；
   - `T3-M5` 共享平滑入口：`withheld_eval.smooth_with_external_prior(...)` 仅基于外部先验旋转矩阵与可选已观测传感器陀螺仪完成 $SO(3)$ 李群 WNOJ 平滑；
   - Task 4（`scripts/spine/calibrate_noise_from_p0.py`、`tests/test_spine_calibration.py`）：提取各被试静态站立（class 3）`LowerBack`/`UpperBack` 去偏残差三轴 `1.4826 * MAD` 与 5-SD 缩尾协方差，按 $9.80665^2$ 缩放计算名义无量纲噪声尺度 $\kappa_{\text{nominal}}=\sigma_a\tau^2/L$ 并补充 WP3 慢模态 5° 精度方向计数，绝不覆盖旧 `results/spine_offline_v1/`。
2. **本地 Codex (`b8b9b53` → `186c975`)**：
   - `T3-M4 / T3-M5`（`src/task3_dl/{rotation_6d.py,model.py,dataset.py,evaluation.py}`、`scripts/task3_dl/train_bigru_6d.py`、`tests/test_task3_bigru_6d.py`）：面向云端 RTX 4090 的传感器屏蔽感知双向 GRU（17 节点 × 13 维局部标准化惯性/旋转前两列/观测标记 = 221 维），被留出通道零泄露编码，输出连续 6D 正交旋转与测地线损失；`--apply-wnoj-post` 无缝接入 VM `withheld_eval.smooth_with_external_prior`（仅对预测旋转做 `T3-M5` 后平滑，绝不读取留出陀螺）；
   - Task 1/2 论文图表与分层汇总（`scripts/analysis/build_task12_paper_figures.py`、`tests/test_paper_analysis_scripts.py`）：英文坐标出版级图表（5 张 `.png`）与严谨隐私脱敏（抑制单人 B 组及互补 C 组均值以防逆推）、官方试次边界分位点与窗口纯度错误分析。
3. **指挥端联合整备与 VM 终验排错 (`eeaa9f7` → `d4fe5c3` → `d0532f2`)**：
   - 更新 `.gitignore` 精确放行 `spine_p0_calibration/*.csv`、`task3_v4_accel_g/**/*.csv`、`task3_bigru_v4/**/*.csv`、`task12_paper_figures/*.csv`，同步屏蔽所有 `_parts/`、`_checkpoints/` 与逐人 `per_subject_metrics.csv`；
   - 新增无侵入接续总控 `scripts/run_cloud_task3_task4_v4.sh`（默认云端 RTX 4090 BiGRU 参数 `--epochs 40 --patience 6 --samples-per-epoch 1024 --train-frames 240 --device auto --apply-wnoj-post`），扩展 `scripts/package_cloud_v4_bundle.py`；
   - Codex 复现并由 VM-Claude 修复 `run_cloud_task3_task4_v4.sh` 中 `--resume` / `--overwrite` 互斥冲突与 `_parts` 打包过滤（`d4fe5c3`）。

---

## 二、联合验证记录

| 环境与范围 | 实测结论 |
| :--- | :--- |
| Linux VM 全仓 Pytest（Python 3.12 + `pyarrow 26.0.1` + `torch 2.15.0+cpu` + `lightgbm`） | **`101 passed, 1 skipped (CUDA device required), 0 failed`**（102 collected；含 `test_diag_p0` 7、`test_paper_analysis_scripts` 3、`test_spine_calibration` 3、`test_spine_offline` 37、`test_task3_bigru_6d` 7/8 [含 `[parquet]`]、`test_task3_wnoj` 30、`test_wp1_v4_features_and_masking` 14） |
| Windows 宿主机（Python 3.14 + RTX 5060 CUDA + `PyTorch 2.11.0+cu128`） | `test_bigru_cuda_forward_backward_smoke` 通过（`1.237281 -> 0.719538 rad`），合入后复测 CUDA 烟测（`1.1769 -> 0.7972 rad`）与 CLI `['T3-M4', 'T3-M5'] cuda 1.34s` 通过 |
| 真实 Task 1（`main@17bb054`）出图实测 | 生成 `fig_task1_ablation_raw_vs_hmm.png` 与 `fig_task1_others_and_walk3.png`，视觉与 `comparison_summary.json` 数值完全一致 |
| Post-Task-2 总控 `--smoke`（`scripts/run_cloud_task3_task4_v4.sh --smoke`） | 首跑 `2m02s` 退出码 0（54 文件 ZIP 通过 `testzip`，无 `_parts`/`_checkpoints`）；二次 `[RESUME_SKIP]` `1.1s`；三次 `--overwrite` 完整重算 `2m13s` |
| 原有总控兼容冒烟（`scripts/run_cloud_all_v4.sh --smoke`） | `1m11s` 退出码 0（`ALL_STAGES_COMPLETE`，105 文件 ZIP） |
| 公开差分纯净度审计（`17bb054..d0532f2`） | `git diff` 扫描零内部讨论路径与 AI 角色标识 |
