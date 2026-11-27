# V3.0 核心文献地图（经三轮交叉审计与 DOI 纠偏）

> **使用规范**：本清单所有 DOI/arXiv 均已经过三轮交叉核查与勘误；引用时严格遵守每篇文献的**适用边界**，不将背景/方法参考夸大为对本数据集的直接证明。

---

## 第一梯队：核心理论文献（泛函样条–MAP–旋转流形）

1. **Wahba, G. (1978)**  
   *Improper Priors, Spline Smoothing and the Problem of Guarding Against Model Errors in Regression.*  
   *Journal of the Royal Statistical Society: Series B*, 40(3), 364–372.  
   DOI: `10.1111/j.2517-6161.1978.tb01050.x`  
   * **用途**：光滑样条与贝叶斯后验估计等价性的奠基理论（注意：$\int|\theta''|^2 dt \leftrightarrow$ 2 阶 IWP 三次样条；$\int|\theta'''|^2 dt \leftrightarrow$ 3 阶 IWP 五次样条）。

2. **Kohn, R., & Ansley, C. F. (1987) [SIAM]**  
   *A New Algorithm for Spline Smoothing Based on Smoothing a Stochastic Process.*  
   *SIAM Journal on Scientific and Statistical Computing*, 8(1), 33–48.  
   DOI: `10.1137/0908004`  
   * **用途**：样条平滑转化为状态空间滤波/平滑高效算法的计算依据（已与同年 *Biometrika* 论文区分）。

3. **Ansley, C. F., & Kohn, R. (1986)**  
   *On the Equivalence of Two Stochastic Approaches to Spline Smoothing.*  
   *Journal of Applied Probability*, 23, 391–405.  
   DOI: `10.2307/3214367`  
   * **用途**：随机过程样条平滑等价性的直接数学补充依据。

4. **Ansley, C. F., & Kohn, R. (1987) [Biometrika]**  
   *Efficient Generalized Cross-Validation for State Space Models.*  
   *Biometrika*, 74(1), 139–148.  
   DOI: `10.1093/biomet/74.1.139`  
   * **用途**：状态空间模型平滑超参数选择的 GCV 依据。

5. **Ovrén, H., & Forssén, P.-E. (2019)**  
   *Trajectory Representation and Landmark Projection for Continuous-Time Structure from Motion.*  
   *International Journal of Robotics Research (IJRR)*.  
   DOI: `10.1177/0278364919839765`  
   * **用途**：连续时间样条扩展至 $SO(3)/SE(3)$ 旋转流形的理论参考（注：原文对象为 SfM，不可声称已直接解决人体关节流形优化）。

6. **Zhang, Z., Stringer, A., Brown, P., & Stafford, J. (2024)** *(待核查作者全名，已核 DOI)*  
   *Model-Based Smoothing with Integrated Wiener Processes and Overlapping Splines.*  
   *Journal of Computational and Graphical Statistics*, 33(3), 883–895.  
   DOI: `10.1080/10618600.2023.2289532`  
   * **用途**：高阶积分维纳过程（IWP）与样条平滑的现代统计实现参考。

7. **Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019)**  
   *On the Continuity of Rotation Representations in Neural Networks.*  
   *CVPR 2019*.  
   * **用途**：神经网络回归姿态采用 6D 连续旋转表征的依据（物理优化仍回到 $SO(3)$ 误差状态）。

---

## 第二梯队：稀疏重建、接触锚点与表征方法文献

8. **Huang, Y., Kaufmann, M., Aksan, E., Black, M. J., Hilliges, O., & Pons-Moll, G. (2018) [DIP]**  
   *Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time.*  
   *ACM Transactions on Graphics (SIGGRAPH Asia)*, 37(6), Article 185.  
   DOI: `10.1145/3272127.3275108`  
   * **用途**：6-IMU 稀疏人体姿态重建与双向 RNN 时序先验范式参考。

9. **Yi, X., Zhou, Y., & Xu, F. (2021) [TransPose]**  
   *TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors.*  
   *ACM Transactions on Graphics (SIGGRAPH)*, 40(4).  
   * **用途**：多阶段稀疏姿态估计与足部接触概率辅助全局根速度估计参考（需按活动扩展为多接触锚点）。

10. **Yi, X., Zhou, Y., Habermann, M., Shimada, S., Golyanik, V., Theobalt, C., & Xu, F. (2022) [PIP]**  
    *Physical Inertial Poser (PIP): Physics-Aware Real-Time Human Motion Tracking From Sparse Inertial Sensors.*  
    *CVPR 2022*, 13167–13178.  
    DOI: `10.1109/CVPR52688.2022.01282` *(已纠偏，原误作 `.01280`)* | arXiv: `2203.08528`  
    * **用途**：运动学网络估计 + 物理约束优化精修的范式参考。

11. **Skog, I., Händel, P., Nilsson, J.-O., & Rantakokko, J. (2010)**  
    *Zero-Velocity Detection—An Algorithm Evaluation.*  
    *IEEE Transactions on Biomedical Engineering*, 57(11), 2657–2666.  
    DOI: `10.1109/TBME.2010.2060723`  
    * **用途**：足部零速检测的似然比检验框架（引用时沿用原论文具体检测器术语）。

12. **Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019)**  
    *Similarity of Neural Network Representations Revisited.*  
    *ICML 2019*, PMLR 97, 3519–3529. arXiv: `1905.00414`  
    * **用途**：Linear CKA 衡量传感器表征相似性（仅作解释，不替代完整屏蔽重训练）。

13. **Lundberg, S. M., & Lee, S.-I. (2017)**  
    *A Unified Approach to Interpreting Model Predictions.*  
    *NeurIPS 2017*. arXiv: `1705.07874`  
    * **用途**：SHAP 模型贡献分配与交互分析参考（注意交互符号依赖价值函数与特征相关性）。

---

## 第三梯队：硬件基准、步态无量纲化与 Task 4 脊柱生物力学文献

14. **Choo, C. Z. Y., Chow, J. Y., & Komar, J. (2022) [关键外部硬件基准]**  
    *Validation of the Perception Neuron system for full-body motion capture.*  
    *PLOS ONE*, 17(1), e0262730.  
    DOI: `10.1371/journal.pone.0262730` | PMID: `35061781` | PMCID: `PMC8782534`  
    * **核心实证**：10 名受试者、17 单元 Perception Neuron vs 120 Hz 8 相机 VICON（基于 BVH 关节角）：膝屈伸平均 RMSE `2.66°`（r = `0.93`）、慢跑踝 RMSE `4.47°`、静止行走踝平均偏置 `22.74°`，肩外展/内收 RMSE 偏高。  
    * **引用边界**：仅作 Perception Neuron 系统级精度背景与未来光学动捕实验设计依据，不直接当作本数据集 4° 误差上限，亦不将 LOA > 95% 误解为误差 < 5%。

15. **Hof, A. L. (1996)**  
    *Scaling gait data to body size.*  
    *Gait & Posture*, 4(3), 222–223.  
    DOI: `10.1016/0966-6362(95)01057-2` *(已纠偏，原误作 `-G`)*  
    * **用途**：人类步态数据按腿长 $L$ 与重力 $g$ 无量纲归一化（如无量纲步频 $f\sqrt{L/g}$）的直接依据。

16. **Alexander, R. M., & Jayes, A. S. (1983)**  
    *A Dynamic Similarity Hypothesis for the Gaits of Quadrupedal Mammals.*  
    *Journal of Zoology*, 201(1), 135–152.  
    DOI: `10.1111/j.1469-7998.1983.tb04266.x`  
    * **用途**：Froude 数动态相似理论的经典源头（原文针对四足动物，人类步态直接引 Hof 1996）。

17. **McClintock, F. A., et al. (2024)**  
    *Validity and Reliability of Inertial Measurement Units Used to Measure Motion of the Lumbar Spine: A Systematic Review of Individuals With and Without Low Back Pain.*  
    *Medical Engineering & Physics*, 126, 104146.  
    DOI: `10.1016/j.medengphy.2024.104146`  
    * **用途**：305 名受试者系统综述，建议在 **L1 与骶骨** 等腰椎解剖边界布设 IMU 以减少胸椎运动串扰，直接支撑 Task 4 的 L1 选址。

18. **Beange, K. H. E., Chan, A. D. C., & Graham, R. B. (2024)**  
    *Investigating Concurrent Validity of Inertial Sensors to Evaluate Multiplanar Spine Movement.*  
    *Journal of Biomechanics*, 164, 111939.  
    DOI: `10.1016/j.jbiomech.2024.111939`  
    * **用途**：C7、T12、S1 多位置 IMU 与光学动捕的同步多平面效度验证，直接支撑 Task 4 脊柱验证实验协议。

> **下一阶段待补文献**：脊柱多节段运动学与硬拉腰椎屈曲损伤阈值原始研究、柔性脊柱曲率传感器原始实测论文、以及 1–2 篇攀岩仪器化岩点测力论文（供 Task 4 拓展讨论）。
