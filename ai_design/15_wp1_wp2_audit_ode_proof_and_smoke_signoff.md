# 15 — WP1 / WP2 代码审计、WNOJ ODE 符号证明与 Linux 全链路冒烟终审归档

- **归档时间**：2026-11-28 04:09（CST）
- **最终冻结 `origin/main`**：`3e9e5ce7b2538c91df1475144f385b19236a4667`（`3e9e5ce`）
- **审签角色**：网页端 GPT（首版代码审查）+ Antigravity（按总决策人授权接棒终审、ODE 数学实证与集成）+ VM-Claude（真实 Linux 82/82 单测与全链路冒烟实跑）

---

## 一、网页端 GPT 代码审查要点与逐项闭环

1. **P0-1（WNOJ 角速度变换符号复核 —— ODE 实证反转，确认代码正确、仅注释笔误）**：
   - 网页端 GPT 审阅 `wp2-task3-wnoj@bf87198` 时，见 `src/task3/wnoj_prior.py` 第 14 行注释误写为 `J_l^{-1}(xi) ~= I + hat(xi)/2`，遂怀疑代码 `e2 = (w1 - 0.5 * np.cross(xi, w1)) - w0 - dt * a0` 符号相反。
   - **独立 ODE 数值证明**：对 $\dot R = [\omega(t)]_\times R$（非共线 $\omega(t) = \omega_0 + a_0 t$，$\omega_0=[1.2,-0.7,0.5]$，$a_0=[-0.8,1.5,2.0]$）使用 `scipy.integrate.solve_ivp`（`rtol=1e-13, atol=1e-14`）直接积分求 $\xi(t)=\mathrm{Log}(R(t)R_0^\top)$ 与 $\dot\xi(t)$：
     - 精确 $J_l^{-1}(\xi)\omega_1 = (I - \frac{1}{2}[\xi]_\times + c[\xi]_\times^2)\omega_1$ 误差：`7.36e-11`；
     - 一阶 `w1 - 0.5*(xi x w1)`（VM 原代码）误差：`3.49e-3`（严格 $O(\|\xi\|^2)$）；
     - 反号 `w1 + 0.5*(xi x w1)` 误差：`1.17e-1`（增大 **33 倍**，符号错误）。
   - **闭环**：修正 `wnoj_prior.py` 第 14 行注释为 `I - hat(xi)/2`，并在 `tests/test_task3_wnoj.py` 新增 `test_wnoj_kinematic_consistency_against_ode()`（含 `dt` 减半误差精确缩小 `4.0×` 的 $O(\Delta t^2)$ 检验：`1.04e-3 → 2.60e-4 → 6.50e-5 → 1.62e-5`）。
2. **P1（V3 留出校准边界）**：已在 `scripts/task3/run_task3_wnoj.py` 的 `summary.json` 显式写入 `v3_uses_standing_calibration_all_sensors` 与 `v3_calibration_note`，明确区分 V3 衍生指标与严格留出主结果 V1/V2。
3. **三项合入前加固全部关闭**：
   - Sticky HMM `dt_seconds` 改由真实 `stride_frames / 60.0` 读取（`codex/wp1-task12-v4@917786f`）；
   - `parse_fold_subsets()` 新增 17 传感器名白名单、去重与 `len(obs) == k` 强校验及 3 条负向单测；
   - Task 3 默认 `--workers` 下调至 `min(4, os.cpu_count())` 控制首次全量运行峰值 RSS；Task 1/2/3 全部增量统计项与 `ranking_prefix` 对照、`scripts/package_cloud_v4_bundle.py`、`scripts/run_cloud_all_v4.sh` 全部合齐。

---

## 二、真实 Linux 全链路验收实测（Python 3.12 + 真实 `pyarrow` + `lightgbm` + `scikit-learn`）

1. **全仓单元测试**：`pytest -v tests/` → **`82 passed in 59.5s`**（0 failed, 0 skipped：P0 7 + WP1 14 + WP2 24 + WP3 37）。
2. **一键全链路冒烟（Run 1）**：`bash scripts/run_cloud_all_v4.sh --smoke` → **73 秒，`ALL_STAGES_COMPLETE`**，Stage 0–6 全部 `DONE`：
   - Stage 4 实证读取 Stage 3 逐折 `per_fold_selected_subsets.json`（`H3:k3`, `H3:k4` 各折不同）；
   - `accel_unit = "none"`、`accel_factor_registered = false`；求解器收敛率 `1.0`、失败率 `0`。
3. **断点续跑实测（Run 2）**：同目录复跑 → **5 秒，Stage 0–6 全部 `[RESUME_SKIP]`（匹配 `config_sha256`）**。
4. **最终打包 ZIP 实测**：`cloud_v4_final_bundle.zip` 共 **103 文件**（87 个 JSON/CSV/MD），含 `cloud_v4_final_bundle/final_report/rowing_2s_vs_4s.csv`。
5. **纯净度扫描**：`origin/main@3e9e5ce` 全仓纯净扫描 **0 命中**，无 `ai_design/`，无 AI 角色名/本地路径/尾注。