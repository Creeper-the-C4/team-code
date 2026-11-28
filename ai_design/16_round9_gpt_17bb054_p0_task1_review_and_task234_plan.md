# 16 — `main@17bb054` 阶段审签与 Task 2/3/4 下一阶段协同规范

- **归档时间**：2026-11-28 15:56（CST）
- **审签对象**：`origin/main @ 17bb054`（`46bc8a1` 30 人 P0 诊断 + `17bb054` Task 1 V4 全量 48 折实验）
- **审计意见**：网页端 GPT 正式签发 **P0 通过**（正式批准 `g` 单位，`Acc_SI = 9.80665 * Acc_raw`，自由线性加速度、严禁再减重力）、**Task 1 V4 通过（结果冻结）**、**Task 2 启动批准**、**Task 3 WNOJ + BiGRU（`T3-M0` 至 `T3-M5`）+ 局部鲁棒 Acc 因子研发批准**、**Task 4 P0 静态三轴噪声与 $\kappa$ 补充标定批准**。

---

## 一、核心阶段实测基线确认

1. **P0（30 人，172.3 s）**：
   - `diag_accel_unit`：`verdict = "g"`（`T1 -> g`，6,585 腾空帧中位数 `-0.98169`、MAD `0.02459`；`T2 -> g`，上楼抬升 `0.33179 m`、步行步幅 `1.18249 m`）；静立模长中位数 `LowerBack = 0.009897 g`、`UpperBack = 0.012220 g`；`T3` $R^2 = 0.3476$（不足以直接标定全身跨关节杆臂）；`T6` >2000°/s 罕见、冲击峰半高宽 1–2 帧。
   - `diag_orientation_independence`：双膝/双肘未触发 `skeleton_constrained_suspected`；17 传感器 `gyro ~ quat` 斜率 `0.991–1.030`、$R^2 = 0.934\text{–}0.978$；一阶残差自相关与腿段残差相关均接近 0（仍不宣称厂商算法完全硬件独立）。
   - `diag_subject_protocol`：6 折互斥通过；26/28 完整被试步频严格单调；`P02/P03` 非单调、`P07/P25` 缺步行速度类；结构异质性分组（`P01–P24`、`P25`、`P26–P30`）按描述性异质性保留；`rowing→sitting`（873 例）中 `80.53%` 位于试次中间 80% 区间。
2. **Task 1 V4（8 组 × 6 折 = 48 折，107.7 min，220,702 个 2s 窗 / 108,613 个 4s 窗）**：
   - `F3_2s Raw`（`88.57% / 92.15%`）精确复现旧基线；
   - 最高总分：`F1B_2s + Sticky HMM`（被试等权 `90.74%` / 合并 `94.11%` / 12 命名类 `96.59%` / Walk3 `94.44%` / Rowing `99.75%`）；
   - 严格导出通道级独立主线：`F1H_2s + Sticky HMM`（被试等权 `90.22%` / 合并 `93.70%` / Rowing `99.76%`），零四元数下已超过 `F3_2s`（`93.24%`）与 `F4_2s`（`93.38%`）；
   - 论文必须遵守探索性排序、固定 13 类缺类计零规则与离线 HMM 边界声明。

---

## 二、当前并行工程协同安排（云端 Task 2 进行中 + 本地 Task 3/4 升级）

1. **云端**：继续完成 Task 2 完整传感器精简队列（`F1H ranking_prefix` → `F1H exact backward` → `F3 backward` → `F1H forward`）。
2. **本地 VM-Claude（`20261128-019`，`vm/task34-p0-calibrated`）**：
   - 补齐 `T3-M3`（仅对上下有观测邻居的夹心节段启用局部鲁棒加速度因子，`9.80665` 换算 + Huber/Cauchy 降权 + WNOJ 平滑角速度/角加速度），并预留 `T3-M5` 混合后平滑 API；
   - 编写 Task 4 P0 三轴静立残差协方差与 $\kappa_{\text{nominal}} = \sigma_a \tau^2 / L$ 标定脚本（只读输出至独立目录 `results/spine_p0_calibration/`）。
3. **本地 Codex（`20261128-020`，`codex/task3-bigru-and-analysis`）**：
   - 编写专为云端 RTX 4090 GPU 设计的 `Sensor-Dropout BiGRU (6D)`（`T3-M4`/`T3-M5`，20 训练被试拟合标准化与结构化传感器掩蔽、零泄露单测 + RTX 5060 CUDA 测试、读取 Task 2 `per_fold_selected_subsets.json` 执行 H3）；
   - 编写 Task 1/2 聚合脱敏论文图表脚本 `scripts/analysis/build_task12_paper_figures.py`（英文坐标避免字体方框，严禁导出个人民感明细）。