# 2026“认证杯”国际数学建模竞赛（ICM）C 题：人体运动与传感器布置

本仓库（`Creeper-the-C4/team-code`）用于沉淀我们团队在 **2026“认证杯”ICM C 题** 中的完整调研、文献审计、数学建模文档、可复现代码与实验结果。

## 最新进度 2026年11月28日（云端执行顺序与 V4.0 落库状态）

> **云端服务器操作请直接按 [docs/07_v4_engineering_spec_and_cloud_handoff.md](docs/07_v4_engineering_spec_and_cloud_handoff.md) 的第 1–4 步顺序执行**：
> 1. `git pull --ff-only origin main` 拉取最新代码并记录 `git rev-parse HEAD`；
> 2. 若仓库已有 `scripts/run_cloud_all_v4.sh`，直接进 `tmux` 一条命令连跑全量并生成 `cloud_v4_final_bundle.zip`；若尚未合入总脚本，先花 3–10 分钟跑完已入库的 **3 个 P0 只读诊断脚本**（[scripts/diagnostics/README.md](scripts/diagnostics/README.md)）；
> 3. 回传启动时的 `HEAD` 号/日志及跑完后的小体积结果包（`results/p0_diagnostics/` 或 `cloud_v4_final_bundle.zip`），不含大文件；
> 4. 验收 P0 单位裁决与本轮结果后，再推送带物理约束精修版与 GPU 深度模型对照脚本跑最终收官轮。

- **当前已合入 `main` 的模块**：
  - P0 只读实测诊断脚本（`scripts/diagnostics/`，7/7 单测通过）
  - Task 4 脊柱可观性离线包、敏感性扫描与 3 张论文图（`src/spine/` + [results/spine_offline_v1/summary.md](results/spine_offline_v1/summary.md)，37/37 单测通过，合计 44/44 通过）
- **Task 1 LightGBM 六折探索性基线（2026-11-27）** 保持冻结作为对照：

| 基线测试指标（`lightgbm_baseline_v1`） | 结果 |
|---|---:|
| 合并测试 13 类 Macro-F1 | 0.9215 |
| 被试等权 13 类 Macro-F1（主评估补充口径） | 0.8857 |
| 合并测试准确率 | 93.35% |
| 六折 13 类 Macro-F1 均值与标准差 | 0.9206 ± 0.0325 |
| 测试参与者与窗口 | 30 人，220,702 个 2 秒窗 |

入口：[V4.0 云端交接手册](docs/07_v4_engineering_spec_and_cloud_handoff.md) · [P0 诊断说明](scripts/diagnostics/README.md) · [Task 4 脊柱可观性结果](results/spine_offline_v1/summary.md) · [Task 1 基线说明](docs/06_task1_lightgbm_baseline.md) · [数据预处理快照](docs/05_independent_preprocessing_progress.md)

> **关键红线**：加速度单位分歧在云端运行 `scripts/diagnostics/diag_accel_unit.py` 完成实测裁决前，**不得**盲乘 `9.81`、重复去重力或在 Task 3 中启用 SI 单位加速度因子。

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
│   ├── reports/                    # 中文进度报告
│   ├── archive/                    # 本轮更新前的文档归档
│   └── reference_images/
├── scripts/preprocessing/          # 预处理与数据读取脚本
├── scripts/task1/                  # 分类基线训练和结果汇总
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
