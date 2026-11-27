# 独立预处理结果 independent_v1

本目录保存2026年11月27日独立处理产生的结果。尚未训练分类、传感器精简或重建模型。

## 已上传的实际数据结果

- `indices_and_quality.zip`：约2.45 MB，内含30名参与者的真实试次表、2秒和4秒窗口索引（Parquet）、逐通道质量表、逐人质量与特征记录，以及汇总元数据。解压后顶层为 `results/`。
- `reproducibility_handoff.zip`：脚本、试次边界、环境和检查报告，不含大数组。
- `final_report.json`、`final_storage_validation.json`、`access_validation.json`：处理规模与验证结果。
- `subject_summary.csv`、`subject_distribution.csv`、`activity_distribution.csv`：逐人、类别和过渡分布。
- `physics_diagnostics.csv`：抽样四元数导数与世界/体系角速度的拟合诊断。
- `body_dimensions_m.csv`、`body_dimension_schema.json`：米制身体尺寸及来源列映射；原表和3行解剖定义另存。
- `schema.json`、`subject_splits_6fold.json`、`baseline_feature_schema.json`：通道与单位决策、参与者划分和40项特征定义。
- `source_file_manifest.json`：源文件SHA256及与现有ZIP条目的CRC核对结果。

基础特征形状为 `[220702,17,40]`，按30人分别落盘，与对应 `windows_2s.parquet` 行序一致。22,690个低能量主频值保留NaN；不做全体拟合插补。

## 仍在云端的大型文件

约9 GB的连续信号缓存和约600 MB的基础特征数组不进入Git历史，符合仓库大文件约定。云端根目录：

```text
/root/autodl-tmp/C_Problem/data/processed/independent_v1/
└── subjects/P01 至 P30/
    ├── signals.npy               # [帧,17,10] float32
    ├── labels.npy                # 原始标签 int8
    ├── trials.parquet
    ├── windows_2s.parquet
    ├── windows_4s.parquet
    └── baseline_features_2s.npy   # [窗口,17,40] float32
```

通过团队授权的SSH/SFTP访问实例获取大数组，或用仓库脚本从同版本原始数据重建。仓库不保存服务器密码、SSH私钥或个人AI凭据。

## 使用约定

窗口起止为0起半开区间，试次来源是作者的0起闭区间。原始单位和标签保留，未额外去重力或旋转Acc/Gyr。数据加载后才在训练折拟合标准化；测试参与者不能指导特征、阈值或传感器选择。

数据来源：https://doi.org/10.6084/m9.figshare.30234940.v1

论文：https://doi.org/10.1038/s41597-026-06710-9

试次边界：https://raw.githubusercontent.com/FudanBSRL/Comprehensive-IMU-Dataset/main/trial_information.csv
