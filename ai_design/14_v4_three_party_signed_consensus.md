# 14 — V4.0 终局设计规范三方会签归档（Antigravity + VM-Claude + 网页端 GPT）

- **签署时间**：2026-11-28
- **审签基线**：
  - `ai-design-discuss`：`d9ac395e5ff6e22b2643d4f62629bb4c6c325478`（本文件归档为其后续终签记录）
  - `main`（审签时纯净基线）：`2361e33df1829366a8d99373032701bfef129486`
- **三方会签状态**：
  - **Antigravity（指挥端）**：**审签通过**
  - **VM-Claude（Linux 端理论/离线建模）**：**审签通过**（`20261128-006`，`vm/wp3-spine-observability@df88298` + `vm/p0-diagnostics@59b2749`，44/44 pytest 通过）
  - **网页端 GPT（独立科研审计）**：**`V4.0 终局设计规范：网页端 GPT 审签通过`**（第八轮正式签发，B1–B3 及第七轮全部 7 项微调无条件通过）

---

## 一、网页端 GPT 终审裁决原文归档

> **V4.0 终局设计规范：网页端 GPT 审签通过**
>
> **审计基线**：`ai-design-discuss @ d9ac395`，`main @ 2361e33`
>
> **批准事项**：
> 1. B1–B3 全部关闭。
> 2. 第七轮七项微调全部通过（跑步机世界系/皮带系统一、`F1^H` 导出通道级独立措辞、运动学速度估计 $\hat v = 2L\sin(\Delta\theta_{\mathrm{th}}/2)f_{\mathrm{step}}$ 定义、L5/骨盆 gauge 区分、DCT-II 基/Neumann 边界/独立 `rank_tolerance`、$K=4/6/8/10$ 敏感性、$\sigma_{\mathrm{bend}}\in\{0.5^\circ..5^\circ\}$ 与 ROM 先验尺度 $0.5\times/1.0\times/1.5\times$ 敏感性、数值秩/仅测量 5° 精度特征方向数/后验标准差严格分离、`pytest -v` 44/44 原始日志核验通过）。
> 3. 正式授权进入工程执行阶段：
>    - **第一阶段**：云端重启后按顺序执行三个 P0 诊断脚本（`diag_accel_unit.py` → `diag_orientation_independence.py` → `diag_subject_protocol.py`），输出至 `results/p0_diagnostics/`。
>    - **第二阶段**：离线启动 WP2 的 Task 3 $SO(3)$ GP/IWP WNOJ 原型（$SO(3)$ 局部误差状态、WNOJ/IWP 先验、陀螺驱动已观测节段、未观测节段 GP 平滑、MAP 与迭代平滑器数值一致性、稀疏块结构与复杂度验证），**暂不启用依赖真实 SI 加速度的物理因子**。
>    - **第三阶段**：WP3 与 WP2 接口整合。
>    - **第四阶段**：P0 结果、WNOJ 原型与 WP3 统一归档并开展真实数据跨被试验证。
>
> *声明：本次为设计规范审签通过，不代表真实数据物理单位已经完成裁决，也不代表最终比赛论文实验结果已经通过审查。*

---

## 二、后续分支与工程落库约定（按用户指令执行）

1. 本分支（`ai-design-discuss`）继续作为 AI 方案审计与留痕隔离分支，**永不合并进 `main`**。
2. 已通过三方审签且经纯净扫描零 AI 痕迹的工程分支（`vm/p0-diagnostics@59b2749` 与 `vm/wp3-spine-observability@df88298`）合入 `main` 并附人类队友云端交接手册 `docs/07_v4_engineering_spec_and_cloud_handoff.md`，供明早云端服务器直接拉取跑全量 P0 与后续实验。
3. 本地离线立即并行推进两项可提前交付的代码包：
   - **WP2（VM-Claude）**：Task 3 $SO(3)$ Lie-group GP/IWP WNOJ 平滑器原型 + 留出传感器评估器 + 合成夹具单测（`accel_unit` 默认 `disabled`，等待 P0 实测裁决）。
   - **WP1（Codex 纯工程实现）**：严格按 V4.0 冻结特征族规范（`F1^W / F1^B / F1^H / F2 / F2' / F3 / F4` + $\mathrm{Deps}(f)\subseteq S$ 硬断言）编写云端就绪的 Task 1/2 特征提取、Sticky HMM 后处理与 6 折帕累托传感器掩码脚本及合成单测。
