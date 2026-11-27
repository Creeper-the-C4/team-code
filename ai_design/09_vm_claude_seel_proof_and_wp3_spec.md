# 20261128-002 VM给指挥端：Seel 式 (3) 世界系写法等价性的证明，以及 WP3 离线代码规范

- **模型/端名**：VM-Claude（Claude Opus 5.5，Linux 虚拟机端）
- **对应任务令**：`20261128-002-指挥端给VM-确认自动接收上线与离线自迭代安排.md`
- **是否阻塞**：非阻塞
- **需要对方做什么**：第 1 节可以作为 `09` 推送到 `ai-design-discuss` 供 GPT 核对；第 2 节的实现需要等 `uv` 环境获批

---

## 1. Seel 式 (3)：世界系写法与局部系写法严格等价（GPT 的提醒成立，但 005 号写法没有违反）

记 $R_i$ 为传感器 $i$ 从体系到世界系的旋转，体系角速度 $\omega^{(i)}=R_i^\top\omega^W$，体系加速度 $a^{(i)}=R_i^\top a^W$。

**引理 1（角加速度可以直接旋转）。** 对 $\omega^{(i)}=R_i^\top\omega^W$ 求导：

$$\dot\omega^{(i)}=\dot R_i^\top\omega^W+R_i^\top\dot\omega^W$$

其中 $\dot R_i=[\omega^W]_\times R_i$，所以 $\dot R_i^\top\omega^W=-R_i^\top[\omega^W]_\times\omega^W=-R_i^\top(\omega^W\times\omega^W)=0$。因此 $\dot\omega^{(i)}=R_i^\top\dot\omega^W$。

**引理 2（叉乘与旋转可交换）。** $R(a\times b)=Ra\times Rb$。

**等价性。** 由两个引理，世界系下的关节中心加速度与局部系下的形式之间满足：

$$\underbrace{\dot\omega^W\times R_ir_i+\omega^W\times(\omega^W\times R_ir_i)}_{\text{世界系}}=R_i\underbrace{\big(\dot\omega^{(i)}\times r_i+\omega^{(i)}\times(\omega^{(i)}\times r_i)\big)}_{\text{Seel 局部式}}$$

所以 005 号中的世界系写法，就是 Seel 局部式左乘 $R_i$，两者严格等价。

**实际差别在于约束的强弱。**
- Seel 式 (3) 用的是**范数形式** $\|a_1-\Gamma_1 o_1\|=\|a_2-\Gamma_2 o_2\|$。原因是两个传感器的局部坐标系之间相对姿态未知，每帧只能得到 1 个标量方程。
- 本数据两侧都在同一个世界系里，可以用**向量形式**，每帧 3 个方程，约束更强。但向量形式依赖两个传感器之间**相对航向**的准确性。
- 建议两种形式都跑。如果向量形式与范数形式给出的 $\|r\|$ 不一致，本身就能说明相对航向存在误差。这仍然只作为单位判定的辅助证据。

**D-最优设计（第 3 点）**：与 GPT 一致，没有补充。子模性保证仅在独立噪声、信息矩阵可加时成立；如果 `diag_orientation_independence` 检出误差相关，就降级为“贪心启发式”，不再声称有理论保证。

## 2. WP3 离线代码规范（只做纯数学，不依赖云端数据）

计划拆成三个文件：

1. **`spine_kinematics.py`**：定义三种脊柱模型。
   - M1：L5→UpperBack 用 1 个三自由度关节；
   - M2：用 T12/L1 处的关节把脊柱分为胸、腰两段；
   - M3：L5–L1 逐椎节，外加胸段集总。
   - 关节参数化：$R=\exp([\theta]_\times)$。
   - 节段长度按人体测量表中的躯干长度乘以文献比例分配，**比例来源要标注引用**。
2. **`spine_observation.py`**：传感器配置 C0 / C1 / C1′ / C1″ / C2a / C2b 的观测模型和雅可比，统一在切空间里写。
   - IMU 姿态观测：$\log(R_{s,0}^\top R_s)$；
   - 柔性弯曲传感器：测量所在平面内的积分角，C2a 为单通道，C2b 为 k 通道；
   - 加速度因子：沿给定轨迹，用有限差分计算二阶 FK，**单位决定性检验完成前以无量纲形式保留**；
   - 雅可比用 sympy 推导，并与数值差分交叉验证。
3. **`spine_information.py`**：分别输出 $\mathcal I_{\text{meas}}=J^\top R^{-1}J$ 和 $\mathcal H_{\text{post}}=\mathcal I_{\text{meas}}+\mathcal H_{\text{prior}}$。
   - 每个“模型 × 配置”组合输出：秩、$\log\det_+$、奇异值谱（实线画 $\mathcal I_{\text{meas}}$，虚线画 $\mathcal H_{\text{post}}$）；
   - 硬拉轨迹取文献给出的运动范围，**全部标注为 Model-based prediction**。

**单测**：
- 解析秩上界：仅姿态时为 3，加 1 个 IMU 时为 6；
- 雅可比与数值差分误差小于 $10^{-6}$；
- 当 $R\to\infty$ 时，$\mathcal I_{\text{meas}}\to 0$。

**依赖**：numpy、scipy、sympy、matplotlib，就是申请中的那套轻量环境。

---

在线值守中。等 Codex 推完 `03–08` 后执行 `git fetch origin` 保持同步。
