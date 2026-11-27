# 2026“认证杯”国际数学建模竞赛（ICM）C 题：人体运动与传感器布置

本仓库（`Creeper-the-C4/team-code`）用于沉淀我们团队在 **2026“认证杯”ICM C 题** 中的完整调研、文献审计、数学建模文档、可复现代码与实验结果。

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
├── README.md                              # 本总览说明
├── .gitignore                             # 忽略大体积原始数据/缓存/模型权重与密钥
└── docs/
    ├── 00_problem_statement.md            # 官方赛题原文要点与红线约束
    ├── 01_multi_ai_discussion_pack.md     # 全程多方 AI 讨论原文归档（V1→V3）
    ├── 02_repo_audit_report.md            # 官方 FudanBSRL 仓库一手源码审计报告
    ├── 03_literature_map_v3.md            # 经三轮核验与 DOI 纠偏的 V3.0 文献库
    ├── 04_v3_blueprint_and_wps.md         # V3.0 建模蓝图、数学纠偏与 WP1–WP3 规划
    └── reference_images/                  # 19 张参考示意图（脊柱解剖、论文图例等）
```

> **数据约定**：原始数据集（`dataset.zip`, ~8.79 GB）及二进制切片缓存**不提交至本仓库**，请按 `.gitignore` 规则保存在本地数据目录。
