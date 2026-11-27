# 独立预处理脚本

这些脚本来自本轮已实际运行的独立流程。基础底座和特征不拟合全体标准化，不训练模型。

| 文件 | 用途 |
|---|---|
| independent_preprocess.py | 源文件校验、逐人连续缓存、试次/窗口表和质量诊断 |
| independent_data_access.py | 完整试次或窗口读取、四元数处理、传感器依赖与训练集标准化接口 |
| independent_features.py | 2秒窗口每传感器40项固定特征，无跨传感器依赖 |
| finalize_preprocessing.py | 汇总质量、身体尺寸和处理报告 |
| package_preprocessing.py | 验证最终存储并打包轻量交付物 |
| trial_information.csv | 作者公开试次边界，未修改 |
| requirements-project.txt | 独立项目新增依赖 |
| environment-lock.txt | 云端运行环境完整记录 |

## 在现有云端复现

```bash
PY=/root/autodl-tmp/C_Problem/.venv/bin/python
$PY scripts/preprocessing/independent_preprocess.py   --source /root/autodl-tmp/dataset   --archive /root/autodl-tmp/dataset.zip   --trials scripts/preprocessing/trial_information.csv   --output /root/autodl-tmp/C_Problem/data/processed/independent_v1
$PY scripts/preprocessing/independent_features.py   --root /root/autodl-tmp/C_Problem/data/processed/independent_v1
```

已完成的逐人缓存会跳过。更改处理规则时应更换输出版本目录，避免复用旧完成标记。

`finalize_preprocessing.py` 和 `package_preprocessing.py` 记录了本轮云端的固定项目根目录；搬到其他环境时先调整根目录。打包脚本还会查找项目根下 `scripts/` 内的原运行脚本，可先复制同名脚本到该位置。

底层复用预装PyTorch/CUDA；`requirements-project.txt` 并不单独安装PyTorch。其他机器按实际GPU环境配置PyTorch后，再安装项目依赖。

Task 2请先选定可用传感器，再计算特征或拟合标准化。`fit_acc_gyr_scaler` 的 held_out_subjects 需包含验证与测试参与者，且四元数不做Z-score。静态窗口的未定义主频可保留NaN供LightGBM处理；其他模型如需插补，只在训练数据上拟合。
