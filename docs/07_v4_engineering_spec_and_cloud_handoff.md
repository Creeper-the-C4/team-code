# V4.0 工程规范与云端全量验证交接手册

本文档面向云端计算环境（RTX 4090 / `independent_v1` 30 人全量缓存），按执行顺序给出**可直接复制执行的云端操作步骤**、**结果回传要求**以及 **V4.0 冻结工程口径**。



> **2026年11月28日接续状态**：第一题V4已于12:37:49完成8组、48折训练并通过脚本核验。详见 [完整结果报告](09_task1_v4_full_validation_report.md) 与 [当日安排](08_2026-11-28_experiment_plan_and_progress.md)。第一题本轮进程已退出，完整结果保留在 `/root/autodl-tmp/C_Problem/results/task1_v4_20261128`。
>
> **第二题已完成**。云端完整目录为 `/root/autodl-tmp/C_Problem/results/task2_v4_20261128`。各折布局和运行结果见 [第二题报告](10_task2_v4_sensor_selection_report.md) 与 [主实验布局](../results/task2_v4/run_20261128_full/F1H_backward/per_fold_selected_subsets.json)。第三、四题从下方独立入口接续。
>
> **极速执行总览（当前 `main` 全库已在 Linux 下通过 101/102 单测（含真实 `pyarrow` + CPU `torch`，唯一跳过项为无卡环境的 CUDA 用例），该 CUDA 用例与 `T3-M4/M5` CLI 已在 RTX 5060 GPU 端到端实测通过；Task 3/4 接续总控与原有总控双套 `--smoke` 含首跑 / 断点续跑 / `--overwrite` 重算均已通过）**
> 1. **核对 Task 2 结果和源代码**：确认主实验的 `F1H_backward/per_fold_selected_subsets.json` 可读，再同步最新 `main`。
> 2. **一键启动 Task 3 / Task 4 / 出图联合流水线**：直接执行第五节 `bash scripts/run_cloud_task3_task4_v4.sh --task2-root <Task2目录> --workers 8`（自动依次执行 Task 4 静态噪声标定 → Task 3 WNOJ `none` → Task 3 WNOJ `g` → RTX 4090 BiGRU 6D + `T3-M5` 后平滑 → Task 1/2 英文论文图 → 轻量打包 `cloud_task34_bundle.zip`）。
> 3. **只需提交小体积汇总产物**：`.gitignore` 已精确放行公共汇总 `.csv/.json/.md` 并自动屏蔽 `per_subject_metrics.csv`、`_parts/`、`checkpoints/` 与大文件。

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

### 3.2 Task 3（流形平滑重建 + 深度对比）与 Task 4（脊柱可观性补偿）

- **Task 3（WP2，WNOJ 含 `T3-M0..M3` 与 BiGRU `T3-M4/M5` 已就绪）**：
  - `T3-M0..M2`（`--accel-unit none` 主基线）：基于 17 节点刚体运动学树与 $SO(3)$ 李群局部误差状态 3 阶 IWP/WNOJ 高斯过程平滑器，采用 **Withheld-sensor reference（留出传感器实测参照，V1/V2 主指标，V3 为全传感器站立标定衍生条件指标）**。
  - `T3-M3`（`--accel-unit g` 局部鲁棒加速度因子对比版）：依据 P0 裁决按 `9.80665` 换算 SI 自由加速度（不重复去重力），**严格限定于上下邻居同时被观测的夹心留出节段**，叠加 Huber 降权（$k=1.345$）与冲击帧掩蔽，并与主基线 `T3-M2` 并列报告真实增减。
  - `T3-M4 / T3-M5`（`scripts/task3_dl/train_bigru_6d.py`，面向云端 RTX 4090）：基于传感器屏蔽感知（`Sensor-Dropout` + 221 维通道零泄露掩码）的双向 GRU，输出连续 6D 旋转矩阵正交映射并用测地线损失训练；`--apply-wnoj-post` 调用共享 `smooth_with_external_prior` 进行 $SO(3)$ 李群后平滑（`T3-M5`），绝不读取被留出通道陀螺仪。
- **Task 4（WP3，离线计算 + P0 实测静立噪声标定已就绪）**：
  - 核心代码：`src/spine/`；离线结果与 3 张论文图：`results/spine_offline_v1/`（详见 [results/spine_offline_v1/summary.md](../results/spine_offline_v1/summary.md) 与 [results/spine_offline_v1/README.md](../results/spine_offline_v1/README.md)）；P0 静态噪声标定：`scripts/spine/calibrate_noise_from_p0.py` 输出至 `results/spine_p0_calibration/`。
  - 核心模型预测结论（基于合成硬拉姿态，非实测值）：仅凭 L5 + 上背 IMU（`C0`）无法同时辨识 L1–L5 与 L5/S1 屈曲；单新增 L1 IMU（`C1`）辨识 L1–L5（后验 std `2.57°`）但 L5/S1 不可观，单新增骶骨 IMU（`C1''`）辨识 L5/S1（`2.34°`）但 L1–L5 不可观；全慢模态数 $K\in\{4,6,8,10\}$、柔性弯曲噪声 $\sigma_{\text{bend}}\in\{0.5^\circ..5^\circ\}$ 及 ROM 先验尺度 $0.5\times/1.0\times/1.5\times$ 敏感性均已验证并归档于 `results/spine_offline_v1/sensitivity.json`。
  - P0 静立噪声标定脚本从各被试站立（class 3）片段提取三轴 MAD 缩尾协方差并映射名义无量纲噪声尺度 $\kappa=\sigma_a \tau^2 / L$（严守静态测量下界边界声明，不覆盖旧 `results/spine_offline_v1/`）。

---

## 四、模块清单（已通过 Linux 101/102 + GPU 联合复验）

| 模块 | 路径 | 功能与验证状态 | 云端执行时机 |
| :--- | :--- | :--- | :--- |
| P0 实测诊断脚本 | `scripts/diagnostics/` + `tests/test_diag_p0.py` | **7/7 通过**；已完成 30 人实测（`verdict="g"`） | 已有 `results/p0_diagnostics/`，无需重跑 |
| Task 1/2 V4 特征族、Sticky HMM 与 6 折帕累托脚本 | `src/v4_features/`、`scripts/task1_v4/`、`scripts/task2_v4/` + `tests/test_wp1_v4_features_and_masking.py` | **14/14 通过**（真实 `pyarrow` + `lightgbm`）；Task 1 48 折已归档 | 保持云端当前运行中的 Task 2 跑完即可 |
| Task 3 $SO(3)$ WNOJ 平滑器（`T3-M0..M3` + `T3-M5` 共享接口） | `src/task3/`、`scripts/task3/run_task3_wnoj.py`、`results/task3_offline_v1/` + `tests/test_task3_wnoj.py` | **30/30 通过**；支持 `none` 主线与 `g` 夹心节段对比 | `run_cloud_task3_task4_v4.sh` 自动串联执行 |
| Task 3 6D Sensor-Dropout BiGRU（`T3-M4/M5`） | `src/task3_dl/`、`scripts/task3_dl/train_bigru_6d.py` + `tests/test_task3_bigru_6d.py` | **Linux CPU+`pyarrow` 7/8 通过（CUDA 项跳过），CUDA 烟测与 `T3-M4/M5` CLI 已在 RTX 5060 GPU 通过** | `run_cloud_task3_task4_v4.sh` 自动紧接 WNOJ 在 GPU 执行 |
| Task 4 脊柱可观性与 P0 静立噪声标定 | `src/spine/`、`scripts/spine/calibrate_noise_from_p0.py`、`results/spine_offline_v1/` + `tests/test_spine*.py` | **37 + 3 = 40/40 通过**；静态下界标定与 `M3` 慢模态补充计数 | `run_cloud_task3_task4_v4.sh` 第一阶段自动执行（1–3 分钟） |
| Task 1/2 论文图表与分层汇总 | `scripts/analysis/build_task12_paper_figures.py` + `tests/test_paper_analysis_scripts.py` | **3/3 通过** + 已实测 Task 1 8 组真实数据出图 | `run_cloud_task3_task4_v4.sh` 末尾自动生成至 `results/task12_paper_figures/` |
| Task 3/4 云端接续总控与打包器 | `scripts/run_cloud_task3_task4_v4.sh`、`scripts/package_cloud_v4_bundle.py` | **新旧双套 `--smoke` 通过**（首跑 / `[RESUME_SKIP]` 续跑 / `--overwrite` 重算均退出码 0，ZIP 无 `_parts`） | Task 2 完成后 `git pull` 一键运行 |

---

## 五、Task 2 收尾后的云端一键接续步骤（面向云端队友）

> **重要**：若 Task 2 仍在云端运行中，**请勿中断当前 Task 2 进程**。等待 Task 2 正常跑完并核对输出包含 `F1H_backward/per_fold_selected_subsets.json` 后，按下面命令拉取最新代码并开启接续运行：

```bash
cd /root/autodl-tmp/C_Problem/team-code
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD

# 1) 确认云端虚拟环境含 torch CUDA（用于 BiGRU 6D）与全部基础包
PY=/root/autodl-tmp/C_Problem/.venv/bin/python
$PY -c "import numpy, scipy, pandas, pyarrow, lightgbm, matplotlib, torch; print('CUDA=', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 2) 在 tmux 中一键依次跑完：Task 4 静态标定 -> Task 3 WNOJ (none) -> Task 3 WNOJ (g) -> BiGRU 6D + T3-M5 -> Task 1/2 论文出图 -> 打包
#    如你的 Task 2 输出在独立目录（例如 /root/autodl-tmp/C_Problem/results/task2_v4_20261128），脚本会自动探测，也可显式传 --task2-root
tmux new -s task34_v4
bash scripts/run_cloud_task3_task4_v4.sh \
  --task2-root /root/autodl-tmp/C_Problem/results/task2_v4_20261128 \
  --workers 8
```

若 Task 2 已直接输出在仓库 `results/task2_v4/` 下，可省略 `--task2-root`（脚本按 `results/task2_v4` → `/root/autodl-tmp/C_Problem/results/task2_v4_20261128` → `/root/autodl-tmp/C_Problem/results/task2_v4` 自动定位）；全部跑完后：
- `.gitignore` 已显式允许 `results/{spine_p0_calibration,task3_v4,task3_v4_accel_g,task3_bigru_v4,task12_paper_figures}` 下的汇总 CSV/JSON/MD，且**自动忽略所有 `per_subject_metrics.csv`、`_parts/` 与 `checkpoints/`**；
- 只需将仓库内新增的小体积公开汇总表与图（及 `results/cloud_task34_bundle.zip` 如需备用传输）推送或提交即可，**切勿提交任何原始信号、模型权重、`.parquet` 或逐人明细文件**。
