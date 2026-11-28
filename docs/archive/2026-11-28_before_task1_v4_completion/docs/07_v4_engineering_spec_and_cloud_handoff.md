# V4.0 工程规范与云端全量验证交接手册

本文档面向云端计算环境（RTX 4090 / `independent_v1` 30 人全量缓存），按执行顺序给出**可直接复制执行的云端操作步骤**、**结果回传要求**以及 **V4.0 冻结工程口径**。


> **当前运行状态 2026年11月28日 10:57:10（北京时间）**：P0 与2秒/4秒特征缓存已完成，第一题正在使用独立入口进行8组配置的六折全量验证，快照为 **2/48 折**。结果目录为 `/root/autodl-tmp/C_Problem/results/task1_v4_20261128`，安排与验收要求见 [当日实验计划](08_2026-11-28_experiment_plan_and_progress.md)。
>
> **当前任务运行期间不并行启动下文整套总控。** 第一题独立入口与总控的进程锁不同，不能用总控的防重入锁判断本轮是否在运行。后续先核对独立任务的进程、日志与 `stage_status.json`，验收第一题结果后再选择第二、三题入口。
>
> **P0 单位裁决已完成**：内部物理检查支持自由加速度数值单位为 `g`，见 [P0 结果](../results/p0_diagnostics/README.md)。原缓存保持不变；第三题本版仍使用 `--accel-unit none`，加速度因子的启用仍需完成物理模型核验。

> **极速执行总览（全部代码已合入 `main` 并通过 Linux 82/82 单测与全链路冒烟，直接按顺序操作即可）**
> 1. **拉取最新 `main` 并核对环境**：执行 `git pull --ff-only origin main` 并记录 `git rev-parse HEAD`。
> 2. **一键启动全量无人值守运行**：进入 `tmux new -s v4_run` 并执行 **`bash scripts/run_cloud_all_v4.sh`**，自动依次完成 **Stage 0（P0 数据检查，3–10 min）→ Stage 1（2s/4s 特征缓存）→ Stage 2（Task 1 全消融 + Sticky HMM）→ Stage 3（Task 2 17 传感器帕累托，含 `ranking_prefix` 对照与 exact wrapper 主线）→ Stage 4（Task 3 WNOJ 留出重建，强制 `--accel-unit none`）→ Stage 6（自动打包 `results/cloud_v4_final_bundle.zip`）**。
>    - 如只想先单跑第一步 P0 数据检查，仍可直接运行下方第一节的 3 条命令并回传 `results/p0_diagnostics/`。
> 3. **只需回报两次**：① 刚启动时发回 `git rev-parse HEAD` 与开头若干行启动日志；② 全部跑完后回传 `results/cloud_v4_final_bundle.zip`（只含小体积表格/图/日志，不传大文件）。
> 4. **后续接力**：拿到本轮结果并完成 P0 加速度单位裁决与验收后，再推送带物理约束精修版及 4090 GPU 深度对照脚本，做最后一轮收官运行。

---

## 一、服务器重启后第一步：执行 P0 实测诊断（约 3–10 分钟，无需 GPU）

P0 三个脚本均为**只读诊断**，仅通过 `scripts/preprocessing/independent_data_access.py` 读取现有 `independent_v1` 缓存（其中 `diag_accel_unit.py` 的 T6 会顺序扫描全部 30 人约 1,345 万帧 / 9 GB 缓存，耗时主要取决于磁盘顺序读速度，约 3–10 分钟），绝不修改任何预处理数据，也不依赖 GPU。

### 1.1 拉取最新代码与环境核对

```bash
cd /root/autodl-tmp/C_Problem/team-code
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD

# 环境检查（云端 .venv 需具备 numpy/scipy/pandas/pyarrow/scikit-learn/lightgbm/matplotlib）
PY=/root/autodl-tmp/C_Problem/.venv/bin/python
$PY -c "import numpy, scipy, pandas, pyarrow, sklearn, lightgbm, matplotlib; print('ENV_OK')"
```

### 1.2 依次运行三个 P0 诊断脚本（或直接跑一键总脚本）

> 如果 `ls scripts/run_cloud_all_v4.sh` 已存在，直接执行：
> ```bash
> tmux new -s v4_run
> bash scripts/run_cloud_all_v4.sh
> ```
> 该总脚本会自动先跑以下 3 个 P0 脚本并继续串联全部任务，且自带 `flock` 防重入与断点续跑（中途意外断开只需重新运行同一条命令）。若尚未合入总脚本，请直接运行下方 3 条命令：

```bash
PY=/root/autodl-tmp/C_Problem/.venv/bin/python
DATA=/root/autodl-tmp/C_Problem/data/processed/independent_v1
OOF=/root/autodl-tmp/C_Problem/results/task1_lightgbm_baseline_v1/oof_predictions.parquet
OUT=results/p0_diagnostics

mkdir -p "$OUT"

# P0-1: 加速度语义（自由加速度 vs 比力）与物理单位（g vs m/s^2）裁决
$PY scripts/diagnostics/diag_accel_unit.py \
    --data-root "$DATA" \
    --out "$OUT"

# P0-2: 四元数独立性检验（单传感器解算 vs 全身骨架约束）与陀螺仪/四元数误差相关性
$PY scripts/diagnostics/diag_orientation_independence.py \
    --data-root "$DATA" \
    --out "$OUT"

# P0-3: 被试组异质性（P01-P24 / P25 / P26-P30）、六折划分审计、标签持续时长 tau、
#       步行步频与简化运动学速度校验、rowing->sitting 误差试次内定位
$PY scripts/diagnostics/diag_subject_protocol.py \
    --data-root "$DATA" \
    --out "$OUT" \
    --splits results/preprocessing/independent_v1/subject_splits_6fold.json \
    --body-dims results/preprocessing/independent_v1/body_dimensions_m.csv \
    --oof "$OOF"
```

> **提示**：如想先用 4 名代表被试快速冒烟测试，可临时加上 `--subjects 1 7 25 27`；正式交付请跑全量（不带 `--subjects`，默认读取全部 30 人）。

### 1.3 回传 P0 诊断结果

脚本仅输出小体积 `.json` / `.md` / `.csv` 报告至 `results/p0_diagnostics/`（不含大体积 `.npy` 或 `.parquet`）。运行完毕后请直接提交推送：

```bash
git status -sb
git add results/p0_diagnostics/
git commit -m "results(diagnostics): add 30-subject P0 diagnostic reports"
git push origin main
```

---

## 二、P0 裁决如何解锁后续实验（硬性红线）

1. **加速度单位未裁决前，禁止在 Task 3 中启用 SI 单位加速度物理因子**
   - `diag_accel_unit.py` 给出三态判定：`g`、`mps2`、`specific_force`（否则 `filtered_or_unresolved`）。
   - 仅当 **T1（跳跃躯干平台）与 T2（零速步幅积分）一致**时才锁定 `g` 或 `mps2`；在 P0 回传裁决前，**不得**对 `Acc` 盲乘 `9.81` 或重复去重力。
2. **跑步机参考系严格统一（禁止混写）**
   - 数据集第 4/5/6 类（慢/中/快走）为固定皮带速度跑步机实验：$v_b \in \{0.9, 1.2, 1.5\}\,\mathrm{m/s}$。
   - **实验室世界系（默认数据系与 Task 3 默认系）**：支撑足速度为 $-v_b$，骨盆平均水平速度 $\approx 0$。
   - **皮带系**：支撑足速度为 $0$，骨盆平均水平速度为 $+v_b$。
   - 简化运动学步速估计 $\hat v = 2L\sin(\Delta\theta_{\mathrm{th}}/2) f_{\mathrm{step}}$（$\Delta\theta_{\mathrm{th}}$ 为单步态周期大腿峰峰摆幅，$f_{\mathrm{step}}$ 为每秒步数，$L$ 为大腿+小腿近似髋踝长）作为协议标称速度参照，不作为独立光学动捕真值。
3. **四元数双覆盖连续化与零泄露底线**
   - 预处理检测出全集存在 `9,604,690` 次四元数符号翻转，任何读取试次/窗口的代码必须经由 `prepare_quaternions` 做试次内半球符号连续化。
   - 所有跨被试验证严格沿用 `results/preprocessing/independent_v1/subject_splits_6fold.json`（每折 20 训练 / 5 内层验证 / 5 外层测试），绝不对外层测试集做早停、标准化或特征筛选。

---

## 三、V4.0 冻结特征族与四任务工程口径

### 3.1 Task 1 & Task 2 特征族（F 系列）

| 特征族 | 构成与坐标口径 | 独立性声明 | 定位 |
| :--- | :--- | :--- | :--- |
| `F1^W` | 单传感器世界系 `Acc/Gyr`（32 维/传感器） | 导出通道级单传感器 | 世界系对照 |
| `F1^B` | 用本传感器自身四元数旋回本体坐标系的 `Acc/Gyr`（32 维/传感器） | 依赖本传感器四元数 | 本体坐标对照 |
| **`F1^H`** | **仅用 `Acc/Gyr` 构造水平模长 + 垂直 Y 分量（24 维/传感器，零四元数）** | **导出通道级独立，航向严格不变**（厂商在全局系输出 Acc/Gyr，不声称原始硬件链路完全独立） | **Task 2 严格主基线** |
| `F2` | 单传感器中心姿态 6D 旋转 + 相邻帧测地角增量（8 维/传感器） | 依赖本传感器四元数 | 姿态增强对照 |
| `F2'` | 航向不变的旋转矩阵第 2 行（重力方向投影）+ 测地角增量（5 维/传感器） | 依赖本传感器四元数，航向不变 | 航向不变姿态对照 |
| `F3` | `F1^W + F2`（40 维/传感器，对应已跑完的 `baseline_features_2s.npy`） | 单传感器 | 已有基线对齐 |
| `F4` | `F3` + 跨传感器相对姿态/运动学组合特征 | 严格满足 $\mathrm{Deps}(f)\subseteq S$ | 强识别上限（受屏蔽集约束） |

- **Task 1 评估指标**：主表并列报告 **被试等权 13 类 Macro-F1**（基线 `0.8857`）与 **合并池化 13 类 Macro-F1**（基线 `0.9215`），同步报告 12 命名类、步行三速子集与 `Others` 混淆分析；Sticky HMM 转移参数 $p_{\text{stay}}=\exp(-\Delta t/\tau)$ 仅由内层训练集估计。
- **Task 2 传感器精简**：以 **`F1^H`（Acc/Gyr only）** 为主结果曲线，绘制 $k=1..17$ 帕累托前沿，并与 `F3`/`F4` 对照；每折独立输出入选传感器子集 $S_k$ 供 Task 3 留出验证复用，防止 Task 2→3 跨折泄露。

### 3.2 Task 3（流形平滑重建）与 Task 4（脊柱可观性补偿）

- **Task 3（WP2，本地离线开发中，即将合入 `main`）**：基于 17 节点刚体运动学树与 $SO(3)$ 李群局部误差状态白噪声加加速度先验（WNOJ / 3 阶 IWP 高斯过程平滑器），采用 **Withheld-sensor reference（留出传感器实测参照，V1/V2/V3 分级）** 验证；加速度因子受 `--accel-unit` 开关门控，待 P0 裁决后再启用。
- **Task 4（WP3，已完成离线理论计算并合入 `main`）**：
  - 核心代码：`src/spine/`；离线结果与 3 张论文图：`results/spine_offline_v1/`（详见 [results/spine_offline_v1/summary.md](../results/spine_offline_v1/summary.md) 与 [results/spine_offline_v1/README.md](../results/spine_offline_v1/README.md)）。
  - 核心模型预测结论（基于合成硬拉姿态，非实测值）：仅凭 L5 + 上背 IMU（`C0`）无法同时辨识 L1–L5 与 L5/S1 屈曲；单新增 L1 IMU（`C1`）辨识 L1–L5（后验 std `2.57°`）但 L5/S1 不可观，单新增骶骨 IMU（`C1''`）辨识 L5/S1（`2.34°`）但 L1–L5 不可观；全慢模态数 $K\in\{4,6,8,10\}$、柔性弯曲噪声 $\sigma_{\text{bend}}\in\{0.5^\circ..5^\circ\}$ 及 ROM 先验尺度 $0.5\times/1.0\times/1.5\times$ 敏感性均已验证并归档于 `results/spine_offline_v1/sensitivity.json`。
  - P0 回传后，将用真实数据的加速度噪声水平标定 WP3 中 $\kappa$ 的实际工作区间。

---

## 四、当前已全部合入 `main` 并通过 Linux 全链路验证的模块清单

| 模块 | 路径 | 当前状态 | 云端执行时机 |
| :--- | :--- | :--- | :--- |
| P0 实测诊断脚本 | `scripts/diagnostics/` + `tests/test_diag_p0.py` | **已合入 `main`（7/7 通过）** | `run_cloud_all_v4.sh` Stage 0 自动执行（3–10 分钟） |
| Task 1/2 V4 特征族、Sticky HMM 与 6 折帕累托脚本 | `src/v4_features/`、`scripts/task1_v4/`、`scripts/task2_v4/` + `tests/test_wp1_v4_features_and_masking.py` | **已合入 `main`（Linux 14/14 全通过，含真实 `pyarrow` + `lightgbm`）** | `run_cloud_all_v4.sh` Stage 1–3 自动执行 |
| Task 3 $SO(3)$ WNOJ 平滑器与留出评估器 | `src/task3/`、`scripts/task3/`、`results/task3_offline_v1/` + `tests/test_task3_wnoj.py` | **已合入 `main`（24/24 通过，含 ODE 运动学校验与双层加速度门控）** | `run_cloud_all_v4.sh` Stage 4 以 `--accel-unit none` 自动执行；待 P0 裁决后再启加速度精修版 |
| Task 4 脊柱可观性分析 | `src/spine/` + `results/spine_offline_v1/` + `tests/test_spine_offline.py` | **已合入 `main`（37/37 通过）** | 离线完成；P0 回传后回填实测噪声区间 $\kappa$ |
| 一键云端连跑总控与打包器 | `scripts/run_cloud_all_v4.sh`、`scripts/package_cloud_v4_bundle.py` | **已合入 `main`（Linux `--smoke` 全阶段 + 二次 `[RESUME_SKIP]` 断点续跑 + 103 文件 ZIP 验证通过）** | 开机 `tmux` 一键启动，结束后一次性回传 `results/cloud_v4_final_bundle.zip` |
