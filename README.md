# 2026“认证杯”国际数学建模竞赛（ICM）C 题：人体运动与传感器布置

本仓库（`Creeper-the-C4/team-code`）用于沉淀我们团队在 **2026“认证杯”ICM C 题** 中的完整调研、文献审计、数学建模文档、可复现代码与实验结果。

## 最新进度 2026年11月28日 第一题 V4 全量验证完成

第一题本轮于 **12:37:49（北京时间）** 完成8组配置、48折训练，30人测试覆盖及核心指标重算核验通过。完整结果见 [第一题 V4 报告](docs/09_task1_v4_full_validation_report.md) 和 [汇总数据](results/task1_v4/run_20261128_full/README.md)。

| 指标 | 旧基线 F3 原始 | 本轮 F1B 2秒 加离线HMM |
|---|---:|---:|
| 被试等权13类 Macro-F1 | 88.57% | 90.74% |
| 合并窗口13类 Macro-F1 | 92.15% | 94.11% |

本轮F3原始预测复现了旧基线得分，8组配置的两项主指标均从离线HMM中获益；F1B加HMM在本轮两项主指标上最高。该排序基于同一套六折的探索性对照；HMM使用后续窗口和已知试次边界，不直接代表实时识别性能。

| 工作 | 实际状态 |
|---|---|
| 数据与P0诊断 | 已完成；自由加速度数值单位支持为g |
| 第一题 V4 | 8/8组、48/48折完成；48个模型与预测保留云端，进入错误分析与论文图表整理 |
| 第二题与第三题 | 第二题正在云端运行中；第三题精修版（WNOJ `none`/`g` 夹心对比 + RTX 4090 6D BiGRU `T3-M4/M5`）、Task 1/2 论文图表脚本与接续总控已全部就绪并合入 `main` |
| 第四题 | 离线可观性与敏感性已完成；新增 P0 静立三轴噪声下界标定脚本 `calibrate_noise_from_p0.py` 待云端接续运行并汇入论文 |
| 存储清理 | 已删除解压后的dataset.zip，释放约8.79 GB；31个CSV、30人缓存及模型保留，剩余约15.07 GiB |

仓库当前为公开状态。本次新发布仅包含汇总指标、分类结果、各折汇总、参数和复现报告；逐人明细、原始信号、逐窗口预测与模型保留云端。详见 [发布清单](results/task1_v4/run_20261128_full/publication_manifest.json)。今日安排见 [实验计划](docs/08_2026-11-28_experiment_plan_and_progress.md)，Task 2 完成后的接续命令见 [交接手册](docs/07_v4_engineering_spec_and_cloud_handoff.md) 第五节。

- **代码交付状态（Linux 全仓 101 通过 / 1 CUDA 跳过 + GPU 单独实测通过；Post-Task-2 接续与原有双套 `--smoke` 首跑 / 断点续跑 / `--overwrite` 均通过）**：
  - **P0 只读实测诊断**：`scripts/diagnostics/`（7/7 通过；30 人实测完成，`verdict="g"`）
  - **Task 1 & Task 2 V4 特征族、Sticky HMM 与 6 折帕累托**：`src/v4_features/`、`scripts/task1_v4/`、`scripts/task2_v4/`（14/14 通过，含真实 `pyarrow` + `lightgbm` 端到端测试）
  - **Task 3 $SO(3)$ 李群 WNOJ 与 6D BiGRU 重建**：`src/task3/`、`scripts/task3/`、`src/task3_dl/`、`scripts/task3_dl/train_bigru_6d.py`、`results/task3_offline_v1/`（WNOJ `30/30` 通过，含 `T3-M3` 夹心 `g` 加速度因子与 `T3-M5` 共享接口；BiGRU Linux CPU+`pyarrow` `7/8` 通过，CUDA 与 `T3-M4/M5` CLI 已在 RTX 5060 GPU 通过）
  - **Task 4 脊柱可观性与 P0 静立噪声标定**：`src/spine/`、`scripts/spine/calibrate_noise_from_p0.py` + [results/spine_offline_v1/summary.md](results/spine_offline_v1/summary.md)（37 + 3 = 40/40 通过）
  - **Task 1/2 论文图表与云端接续总控**：`scripts/analysis/build_task12_paper_figures.py`（3/3 通过 + 已在 Task 1 实测结果上出图核验）、`scripts/run_cloud_task3_task4_v4.sh`、`scripts/run_cloud_all_v4.sh`、`scripts/package_cloud_v4_bundle.py`
- **Task 1 LightGBM 六折探索性基线（2026-11-27）** 保持冻结作为对照：

| 基线测试指标（`lightgbm_baseline_v1`） | 结果 |
|---|---:|
| 合并测试 13 类 Macro-F1 | 0.9215 |
| 被试等权 13 类 Macro-F1（主评估补充口径） | 0.8857 |
| 合并测试准确率 | 93.35% |
| 六折 13 类 Macro-F1 均值与标准差 | 0.9206 ± 0.0325 |
| 测试参与者与窗口 | 30 人，220,702 个 2 秒窗 |

入口：[完整结果报告](docs/09_task1_v4_full_validation_report.md) · [今日实验安排](docs/08_2026-11-28_experiment_plan_and_progress.md) · [第一题运行快照](results/task1_v4/run_20261128_snapshot/README.md) · [V4.0 云端交接手册](docs/07_v4_engineering_spec_and_cloud_handoff.md) · [P0 诊断说明](scripts/diagnostics/README.md) · [Task 4 脊柱可观性结果](results/spine_offline_v1/summary.md) · [Task 1 基线说明](docs/06_task1_lightgbm_baseline.md) · [数据预处理快照](docs/05_independent_preprocessing_progress.md)

> **加速度诊断已完成**：P0 内部物理一致性检查支持已经去重力的自由加速度、数值单位为 `g`；这不是外部仪器校准。原缓存保持原值，需要 SI 单位时在派生计算中乘 `9.80665`，不重复去重力。第三题同时保留无加速度主基线 `T3-M2`（`--accel-unit none`）与夹心节段局部鲁棒加速度对比版 `T3-M3`（`--accel-unit g`），并加入 6D Sensor-Dropout BiGRU 对比版（`T3-M4/T3-M5`）。

## 核心研究主线（V3.0）
**人体运动的多尺度可观测性与最小传感器设计**（$\mathrm{Measurement} \to \mathrm{Representation} \to \mathrm{Observability} \to \mathrm{Inference}$）：
* **Task 1（活动识别与动力学特征）**：严格修复 2,381 个试次边界与测试集早停泄露，基于 13 类监督分类（LightGBM 主基线 + Sensor-Dropout 轻量时序网络）、粘性 HMM、相对姿态及无量纲步频/Froude 假设检验建立严谨跨被试基线。
* **Task 2（任务相关的信息压缩）**：严格 Nested Subject-wise CV + 前向/后向贪心 Wrapper 屏蔽验证，辅以 Shapley 交互与物理不变量 Linear CKA 解释，揭示“识别最小布局 $\neq$ 重建最小布局”。
* **Task 3（流形连续运动重建与留出验证）**：基于 17 节点刚体运动学树、**Wahba 样条–MAP 对偶理论**（2/3 阶 IWP 与 $SO(3)$ 李代数误差状态稀疏优化/L-BFGS）、多接触锚点软融合，并采用 **Withheld-sensor reference（留出传感器实测参照，V1/V2/V3 分级）** 进行定量验证。
* **Task 4（脊柱内部运动可观性补偿）**：针对 17 单元原布局在上背与 L5 之间缺失脊柱传感节点的结构性盲区，以**杠铃硬拉腰椎屈曲安全监测**为主方案，建立多节段脊柱运动学模型（S1/S2/S3），数值计算新增 L1/T12 传感器前后的雅可比秩增量 $\Delta r$ 与奇异值谱变化，递进论证压力鞋垫扩展与独立光学动捕/测力台验证实验，并以室内攀岩闭链静不定作为拓展讨论。

---

## 仓库目录结构

```text
team-code/
├── README.md
├── .gitignore
├── docs/
│   ├── 00_problem_statement.md
│   ├── 01_multi_ai_discussion_pack.md
│   ├── 02_repo_audit_report.md
│   ├── 03_literature_map_v3.md
│   ├── 04_v3_blueprint_and_wps.md
│   ├── 05_independent_preprocessing_progress.md
│   ├── 06_task1_lightgbm_baseline.md
│   ├── 07_v4_engineering_spec_and_cloud_handoff.md
│   ├── 08_2026-11-28_experiment_plan_and_progress.md
│   ├── 09_task1_v4_full_validation_report.md
│   ├── reports/                    # 中文进度报告
│   ├── archive/                    # 本轮更新前的文档归档
│   └── reference_images/
├── scripts/preprocessing/          # 预处理与数据读取脚本
├── scripts/task1/                  # 分类基线训练和结果汇总
├── results/p0_diagnostics/         # 30人P0真实诊断
├── results/task1_v4/run_20261128_full/ # 第一题8组全量汇总与报告
├── results/task1_v4/run_20261128_snapshot/ # V4运行进度与核验快照
├── results/task1/lightgbm_baseline_v1/ # 六折指标 图表 配置
└── results/preprocessing/independent_v1/
    ├── README.md
    ├── indices_and_quality.zip     # 30人真实试次/窗口索引和质量结果
    ├── reproducibility_handoff.zip # 脚本 元数据 环境记录 报告
    ├── final_report.json
    └── ...                        # 分布表 物理诊断 身体尺寸和模式定义
```

> **数据约定**：原始数据集（`dataset.zip`, ~8.79 GB）及二进制切片缓存**不提交至本仓库**，请按 `.gitignore` 规则保存在本地数据目录。

小体积的结果表和索引交付包已按路径显式纳入版本管理；原始数据、约9 GB的连续缓存、约600 MB的特征数组仍保留在云端。旧README及旧忽略规则归档于 `docs/archive/2026-11-27_before_independent_preprocessing`。

本轮更新前的首页、预处理说明和忽略规则归档于 `docs/archive/2026-11-27_before_task1_baseline`。预处理原始结果及第一次归档保留。

2026年11月28日本次进度更新前的首页与云端交接手册归档于 `docs/archive/2026-11-28_before_task1_v4_progress`。

本次完成结果发布前的本地文档归档于 `docs/archive/2026-11-28_before_task1_v4_completion`。
