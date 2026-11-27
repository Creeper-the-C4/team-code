# 官方代码库 `FudanBSRL/Comprehensive-IMU-Dataset` 一手源码审计报告

> **审计对象**：`https://github.com/FudanBSRL/Comprehensive-IMU-Dataset`（`extract_features.py`、`train_base_traditional_models.py`、`train_base_nn.py`、`trial_information.csv`）
> **审计日期**：2026-11-27（经 Antigravity 直接拉取源码核查 + ChatGPT 第三轮逐行交叉复核）

---

## 一、四项核心实锤发现

### 1. 试次边界（`trial_information.csv`）在全部官方基线中被忽略
* **核查事实**：
  * `trial_information.csv` 共 **2,382 行**（含表头），即 **2,381 条独立试次边界记录**（列：`Participant ID, Start Index, End Index`）。
  * 但在官方三份脚本中：
    * `extract_features.py`（L238–244）
    * `train_base_traditional_models.py`（L218–227）
    * `train_base_nn.py`（L161–167）
  * 全部直接使用 `n_timesteps = len(data)` 对整份个人 CSV 滑动切窗，**并未读取 `trial_information.csv`**，仅做了 `if len(activity_labels) > 1 or activity_labels[0] == 0: continue` 的标签过滤。
* **潜在问题**：若同一参与者相邻两个独立试次恰好具有相同活动标签，跨试次边界的非法窗口会绕过混合标签过滤，直接违反赛题“不得跨试次拼接时间序列”的硬性规定。
* **我方修正**：**放弃复用原作者任何预生成 `.npy` 缓存**，所有窗口严格在 2,381 个试次内部独立切分，并为每个窗口记录 `Participant ID / Trial ID / Start Index / End Index / Activity Label / Window Purity`。

### 2. 跨被试 5 折划分核实与论文文字差异
* **核查事实**：
  * `train_base_traditional_models.py`（L408–409）与 `train_base_nn.py`（L741–742）均执行：
    ```python
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(kfold.split(np.arange(N_PARTICIPANTS)))  # N_PARTICIPANTS = 30
    ```
  * 代码实际为**按参与者索引分组的 5 折跨被试划分（每折 24 人训练 / 6 人测试）**，并非随机窗口划分。
* **待标注的复现协议差异**：原论文正文方法部分提及每折 20 人训练 / 5 人测试，与仓库中 30 人 5 折（24/6）存在人数口径差异。论文中如实说明即可，不武断归咎原论文。

### 3. 重大隐患：`train_base_nn.py` 存在测试集驱动早停与模型选择泄露
* **核查事实**：在 `train_base_nn.py` 的 L644–665 行：
  ```python
  valid_loss, valid_accuracy, ... = evaluate_epoch(model, test_loader, loss_fn, device, task)
  ...
  current_metric = results_df.at[epoch, mnt_metric]  # 默认 mnt_metric='valid_accuracy'
  if current_metric > best_metric:
      best_metric = current_metric
      best_cm = valid_cm
      torch.save(model.state_dict(), best_model_path)
  ```
  其中 `test_loader` 直接来自外层测试参与者。这意味着每个 epoch 的早停判断和最优权重保存均直接受测试集表现驱动，存在**测试集参与模型选择**的乐观偏差风险。
* **我方修正**：执行严格的 **Nested Subject-wise CV** —— 外层测试参与者全程锁死，仅在内层训练参与者的再划分验证集上进行早停、超参调节与 Task 2 传感器选择。

### 4. 四元数通道未参与官方基线 & 0 类/混合窗被跳过
* **核查事实**：
  * `extract_features.py` 仅遍历 `SENSOR_TYPE = ['Acc', 'Gyr']` 与 `AXES = ['X', 'Y', 'Z']`（共 $6 \times 17 = 102$ 通道），未利用 4 通道四元数；
  * `train_base_traditional_models.py`（L214–215）与 `train_base_nn.py`（L157–158）显式删除所有含 `'Quat'` 的列；
  * 官方基线全部跳过了标签 `0`（其他）及跨标签混合窗口。
* **17 个传感器官方标准命名**：
  `LowerBack`, `RightThigh`, `RightShank`, `RightFoot`, `LeftThigh`, `LeftShank`, `LeftFoot`, `UpperBack`, `Head`, `RightShoulder`, `RightUpperArm`, `RightForeArm`, `RightWrist`, `LeftShoulder`, `LeftUpperArm`, `LeftForeArm`, `LeftWrist`。

---

## 二、坐标系核验提醒（动工前第一实测任务）
* 据 2026 数据论文说明，Acc、Gyr 和 Quat 均已表达在经校准建立的标准化全局坐标系中：**X 指向参与者左侧，Y 竖直向上，Z 指向校准时正前方**。
* **红线**：在未通过静立试次实测验证前，**禁止默认使用四元数再次旋转 Acc/Gyr**；静立重力方向检验应重点检查 **Y 轴**（模长约 $9.81\ \text{m/s}^2$ 或 $0$，视是否去重力而定），并对全序列四元数执行双倍覆盖（$q \leftrightarrow -q$）半球连续化。

## 三、推荐四步递进消融实验（量化审计价值）
在相同参与者划分与模型超参下依次对比：
1. 原作者切窗口径复现；
2. 仅修复 2,381 个试次边界（+ 修复 NN 内层独立验证早停）；
3. 修复边界并纳入 `0` 类（13 分类）及独立评估过渡窗；
4. 引入姿态四元数衍生特征（相对旋转流形特征 + 无量纲步频）。
