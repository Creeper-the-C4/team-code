# 12 · 网页端 GPT 第七轮终审归档（B1 通过 · B3 静态全通过 · 最后 7 项微调）

- **来源**：`https://chatgpt.com/s/t_6b09c60469c48191a816bdb99ce80888`（已读取 `ai-design-discuss@7ff80c9` 含 11 号报告、JSON 与 3 张 PNG 原图）
- **总体裁决**：**无重大数学错误、不需重谈 V4.0 架构；B1 已通过，B3 静态信息矩阵/Jacobian/3 张图全部通过；关闭以下 7 项微调后即可终局签字！**

---

## 一、已正式通过部分
1. **B1（Seel 式 (7) + $\mathbf o_i = -\mathbf r_i$ + T1 `specific_force` 第三假设 + 7/7 合成单测）**：**正式通过**（真实数据单位裁决仍待云端实跑，SI 因子继续暂停）。
2. **B2 特征规范（`F1^W/F1^B/F1^H/F2/F2′/F3/F4` + `Deps(f)⊆S` + `F1^H` 导出通道级独立主结果 + 跑步机 `0.9/1.2/1.5 m/s`）**：**特征体系全部通过**。
3. **B3 静态可观性 & 3 张实算图**：16/16 静态配置全部达到理论秩上界 $r_{\max}=\min(3n, 3(N_{\text{distinct}}-1)+N_{\text{bend}})$，6 项元数据齐全，float64 多步长中心差分相对误差达到 $3.37\times 10^{-12}\sim 5.76\times 10^{-10}$，3 张 PNG 原图审阅通过。
4. **White & Panjabi (1990) ROM 数值核实**：与原书二手引用表完全一致（L1–L2 `12°/6°/2°` 至 L5–S1 `17°/3°/1°`）。

---

## 二、阻断终局签字的最后 7 项微调清单
1. **统一 Task 3 跑步机参考系与接触锚点速度**：实验室世界系中 $v_{\text{foot}}^{\text{world}} = -v_b$；跑步机皮带系中 $v_{\text{foot}}^{\text{belt}} = 0$（禁止混写“皮带系 + $-v_b$”，统一锁定实验室世界系 $-v_b$，皮带系则写 $0$）；同步精化 `F1^H` 导出通道独立表述与 $\hat v$ 峰—峰摆幅/每秒步数定义。
2. **修正 WP3 gauge / 骨盆参考姿态表述**：改为“L5 为相对姿态参考节点；骨盆/根节段世界姿态作为干扰/参考变量消元，L5 IMU 本身不能独立分离骨盆姿态与 L5/S1 内部关节角”。
3. **补齐 `trajectory_information.json` 中的 `rank_tolerance` 及余弦慢模态基完整定义**（公式、归一化、边界条件）。
4. **增加慢模态数 `K = 4, 6, 8, 10` 敏感性比较**。
5. **增加柔性弯曲传感器噪声 $\sigma_{\text{bend}} \in \{0.5^\circ, 1^\circ, 2^\circ, 3^\circ, 5^\circ\}$ 敏感性**（基于 Zawawi 2013 `2° resolution`、Kam 2019 `0.13°–0.33°`、Nakamoto 2018 `std 7.47°`、FlexTail 2026 `1.05° 3D 打印模板`，2° 仅作名义设计场景；C2a 收窄为“指定平面内跨度积分曲率/净切线转角”）。
6. **严格区分三项口径与 ROM 名义先验**：区分“结构数值秩 / `measurement-only 5°-precision eigen-directions`（$1/\sqrt{\lambda_i}<5^\circ$）/ 含先验的后验标准差”；ROM 折算标为 **`ROM-derived nominal prior scale`** 并补尺度敏感性。
7. **附带真实 `pytest` 验证日志**至 `ai_design/wp3_results/pytest_verification.log`。
