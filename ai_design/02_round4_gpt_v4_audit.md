# 02 · 网页端 GPT 第四轮独立科研审计归档（V4.0 精修框架）

- **来源**：网页端 GPT 第四轮完整回复（`https://chatgpt.com/s/t_6b09a25c91dc81918416a46f2842a04f`）
- **审阅基线**：基于 `3854072`（30 人独立预处理全量实测诊断）与 VM-Claude 阶段 0 的 6 项深度提案（注：此时尚未审阅队友后续推送的 `2361e33` Task 1 LightGBM 六折基线）。

---

## 一、统一主线升级（V4.0）
* **论文主线**：**多尺度人体运动的可观测性、概率重建与最小传感器设计**
  $$\text{Physical Measurement} \to \text{Information Representation} \to \text{Observability} \to \text{Probabilistic Reconstruction} \to \text{Sensor Design}$$
* **核心思想**：分类最小布局、全身构型重建最小布局、脊柱运动监测最小布局是**三个不同的优化问题**，不强求单一传感器数牺牲物理意义。

---

## 二、六大核心问题逐条审计裁决

### 1. 加速度单位（`m/s²` vs `g`）裁决
* **已确认**：标准化全局系（Y 向上）、加速度已去重力（站立模长中位数 `0.0125`）、角速度强烈支持世界系 `rad/s`（拟合斜率 `~0.91`）。
* **对跳跃与刚体回归两步法的审计**：
  * **批准跳跃腾空平台检验**：在 Activity 11 中跨多躯干节段（`LowerBack / UpperBack / Head / Thighs`）检测具有起跳–腾空–落地时序且持续多帧的候选区间，计算低通后 `median` 与 `MAD`，对比 $H_1: -9.81$ 与 $H_2: -1$；注意头部和 L5 并非严格人体质心，不能只看单一全局最小值。
  * **驳回简单跨关节单刚体回归**：$a_B = a_A + \alpha\times r_{AB} + \omega\times(\omega\times r_{AB})$ 仅对**同一刚体上的两点**成立；跨关节父子节段须升级为含相对关节运动项与科氏项的多刚体递推，并受关节中心/传感器偏移影响。
  * **补充第三项证据**：核查 Perception Neuron / Axis Studio 原始导出字段与单位配置。
* **最终红线**：未得出决定性证据前，标注 `Unit declared by authors: m/s²; empirical validation pending`，**不乘 9.81、不再次去重力、暂停 Task 3 依赖绝对加速度尺度的 SI 物理残差**（允许不依赖绝对量纲的 Task 1 分类）。

### 2. P25–P30 结构异质性与评估协议
* **区分三类异质性**：记录长度（P25 仅 198k 帧）、试次数量（P26–P30 达 159–260 试次）、标签结构（混合窗与 0 类暴跌）。不预设为质量下降，先统计试次长度分布与试次内标签切换率。
* **评估三层指标**：窗口加权 Macro-F1 + **参与者宏平均 Macro-F1**（$F1_{\text{subject}} = \frac{1}{30}\sum_{s=1}^{30} F1_s$，重点报告）+ 分层分布（P01–P24 / P25 / P26–P30，固定 13 类集合并报告覆盖情况）。
* **Sticky HMM 修正**：采用连续时间指数离散化 $p_{\text{stay}} = \exp(-\Delta t/\tau)$（HSMM 作后续对照），仅在训练集估计且**严禁跨试次边界传播**。
* **验证协议正式锁定**：统一复用 `subject_splits_6fold.json`（准确命名：*Six-fold subject-wise outer evaluation with fixed inner validation*，20 训练 / 5 验证 / 5 测试，不称完整 Nested CV；关键结果补 30 折 LOSO 稳健性检查）。Task 1/2/3 必须共用同一份带哈希校验的划分文件。

### 3. `independent_v1` 底座复用边界与特征升级
* **冻结直接复用（`DATA_LAYER_V1`）**：原始 CSV、连续 `[frame, 17, 10]` 缓存、原始标签、`trial_information.csv`、`2s/4s` 试次内窗口索引、`body_dimensions_m.csv`、质量报告与原始四元数。云端剩余磁盘仅 ~11 GiB，禁止复制多套大型窗口数组。
* **必须按 V4.0 重做/新增**：4s 特征与多尺度 `2s+4s` 融合（$M_{2s}, M_{4s}, M_{\text{multi}}$）、绕 **Y 轴** 航向规范化（同步旋转 `Acc/Gyr`，且与原始全局系做对照消融）、相对旋转与相对角速度、低能量主频有效性掩码（22,690 NaN 不填零）、传感器独立性诊断、Sensor Hold-out 数据、脊柱雅可比。
* **佩戴偏差增强**：第一版分类先跑无增强基线；后续三种物理增强（传感器-节段标定偏置、位置偏移 $\Delta r$、单节点航向误差）须分立消融，且位置偏移增强须待 Acc 单位确认后才可启用。

### 4. WP2（GP/IWP WNOJ + $SO(3)$ + H1 新息标定）核准与精修
* **已核实 4 篇核心新文献**：
  1. Barfoot, Tong & Särkkä (RSS 2014), *Batch Continuous-Time Trajectory Estimation as Exactly Sparse Gaussian Process Regression*
  2. Anderson & Barfoot (IEEE IROS 2015, 157–164), *Full STEAM Ahead... on SE(3)*, DOI: `10.1109/IROS.2015.7353368`
  3. Tang, Yoon & Barfoot (IEEE RA-L 2019, 4(2), 594–601), *A White-Noise-on-Jerk Motion Prior... on SE(3)*, DOI: `10.1109/LRA.2019.2891492` | arXiv: `1809.06518`
  4. Bell (SIAM J. Optim. 1994, 4(3), 626–636), *The Iterated Kalman Smoother as a Gauss–Newton Method*, DOI: `10.1137/0804035`
* **术语与理论边界修正**：
  * 准确表述为 *Lie-group Gaussian-process smoothing with a white-noise-on-jerk local error-state prior*（不称 *exact Gaussian Wahba smoother on SO(3)*）；
  * 一次 RTS 扫描仅解当前线性化子问题，迭代收敛得到的是非线性 MAP 的局部最优解。
* **建模与防泄露要求**：
  * 批准**已观测节段由陀螺驱动（世界系 $\dot{R}_i = [\omega_i^W]_\times R_i$，并含偏置与噪声模型），未观测节段由 WNOJ/IWP + 关节拓扑耦合 + ROM 共同约束**；
  * 核查四元数与陀螺的误差相关性，增加 `gyro-only / orientation-only / gyro+orientation` 消融；
  * 批准内层验证集 H1 留出残差/新息标定（最多留 2 个核心超参：$q$ 与 $\lambda_j$），标定用 H1 数据不可再充当独立最终测试成绩；
  * **强制执行 Task 2 $\to$ Task 3 逐折布局防泄露**（第 $k$ 折外层测试仅评估第 $k$ 折内层选出的 $S_k$）。

### 5. WP3（Task 4 脊柱 M1–M3 × C0–C3）核准与精修
* **正式纳入**：`M1`（集总腰椎，简化可执行）/ `M2`（胸腰双段，推荐主模型）/ `M3`（多节段，结构欠约束分析） × `C0`（L5+上背）/ `C1`（+L1）/ `C1′`（+T12）/ `C1″`（+骶骨 IMU，同等单传感器候选）/ `C2a`（L1–L5 局部柔性弯曲）与 `C2b`（多点分布式柔性曲率）/ `C3`（足底压力动力学扩展，不与姿态传感器混排）。
* **核心理论区分（V4.0 亮点）**：必须分别报告**纯测量信息矩阵** $\mathcal{I}_{\text{meas}} = J^\top R^{-1} J$ 与**叠加解剖先验后的后验曲率** $\mathcal{H}_{\text{post}} = \mathcal{I}_{\text{meas}} + \mathcal{H}_{\text{prior}}$，严禁把先验约束误写成传感器带来的结构可观性增益。
* **引用边界**：McClintock et al. (2024) 15 项研究 305 人汇总加权 RMSE `< 2.4°`、ICC `> 0.84`，是跨研究汇总值而非任意设备/动作保证精度；不武断写死柔性传感器 $\Delta r = 0$，应按测量平面与跨度如实分析。

---

## 三、开工训练前 P0 级三项轻量诊断脚本（云端 RTX 4090）
1. **`diag_accel_unit.py`**：跳跃候选腾空平台检测（多躯干节段 `median + MAD`）+ 足/小腿极端峰值持续时间核查 + 导出配置核查 + 多刚体标度检验（无法定论时明确输出 `unresolved`）。
2. **`diag_orientation_independence.py`**：逐试次四元数符号连续化 + 世界/体系角速度拟合 + 膝关节三自由度铰链截断诊断 + 相邻节段与陀螺/姿态残差相关性。
3. **`diag_subject_protocol.py`**：30 人试次长度/P25–P30 分层分布 + 0 类/纯度/试次内真实标签切换率 + 六折互斥与类别覆盖审计 + 试次内 HMM 持续时间统计。

## 四、额外推荐的三项高分拓展实验
1. **单点故障鲁棒性**：$R(S) = \min_{j\in S}\mathrm{MacroF1}(S\setminus\{j\})$；
2. **双任务 Shapley 排序对照**：对比 $\phi_i^{\text{HAR}}$ 与 $\phi_i^{\text{Pose}}$，直接证明识别核心传感器 $\neq$ 重建核心传感器；
3. **D-最优可观性设计**：$\max_{|S|\le K}\log\det(\mathcal{I}(S)+\epsilon I)$。
