# 2026“认证杯”国际数学建模竞赛（ICM）C 题：人体运动与传感器布置

本仓库（`Creeper-the-C4/team-code`）用于沉淀我们团队在 **2026“认证杯”ICM C 题** 中的完整调研、文献审计、数学建模文档、可复现代码与实验结果。

## 最新进度 2026年11月27日

已独立完成30名参与者的数据预处理，当前进入 **Task 1 分类基线准备阶段**。本轮尚未训练模型，下面V3.0内容是研究路线，不能视为已实现的模型或实验结论。

- 13,454,722帧、2,381个官方试次；220,702个2秒窗、108,613个4秒窗。
- 已生成逐人连续缓存、试次与窗口索引、每传感器40项基础特征和全量质量检查。
- 原始信号无NaN或无穷值，试次覆盖完整；2秒窗口数量与队友报告相同，处理细节仍待代码对照。
- 加速度保留原值，单位分歧尚未解决；本轮未乘9.81、未去重力、未对世界系信号重复旋转。

入口：[进度说明](docs/05_independent_preprocessing_progress.md) · [中文PDF报告](docs/reports/2026-11-27_independent_preprocessing_progress.pdf) · [处理结果与实际索引](results/preprocessing/independent_v1/README.md) · [复现脚本](scripts/preprocessing/README.md)

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
│   ├── reports/                    # 中文进度报告
│   ├── archive/                    # 本轮更新前的文档归档
│   └── reference_images/
├── scripts/preprocessing/          # 预处理与数据读取脚本
└── results/preprocessing/independent_v1/
    ├── README.md
    ├── indices_and_quality.zip     # 30人真实试次/窗口索引和质量结果
    ├── reproducibility_handoff.zip # 脚本 元数据 环境记录 报告
    ├── final_report.json
    └── ...                        # 分布表 物理诊断 身体尺寸和模式定义
```

> **数据约定**：原始数据集（`dataset.zip`, ~8.79 GB）及二进制切片缓存**不提交至本仓库**，请按 `.gitignore` 规则保存在本地数据目录。

小体积的结果表和索引交付包已按路径显式纳入版本管理；原始数据、约9 GB的连续缓存、约600 MB的特征数组仍保留在云端。旧README及旧忽略规则归档于 `docs/archive/2026-11-27_before_independent_preprocessing`。
