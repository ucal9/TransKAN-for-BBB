---

# 多模态血脑屏障(BBB)渗透性预测模型

本项目实现了一个多模态的血脑屏障(BBB)渗透性预测模型，结合了多种特征提取方法和先进的深度学习技术。

## 环境设置



1. 创建并激活虚拟环境：
   ```
   conda create -n kan_bbb python=3.8
   conda activate kan_bbb
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 使用方法

### 运行主脚本

使用以下命令运行主脚本：

```
python TransKAN.py [参数]
```

### 命令行参数

- `--data_file`: 输入数据文件路径（默认：'./mutiple fusion/b3db_fused_features64d.csv'）
- `--label_col`: 标签列名称（默认：'BBB+/BBB-'）
- `--augment_factor`: 数据增强因子（默认：2）
- `--resample_strategy`: 重采样策略（默认：'auto'）
- `--random_seed`: 随机种子（默认：42）

### 示例

1. 使用默认参数运行：
   ```
   python TransKAN.py
   ```

2. 使用自定义参数运行：
   ```
   python TransKAN.py --data_file my_data.csv --label_col BBB_label --augment_factor 3
   ```


## 数据格式

输入数据应为CSV格式，包含以下列：
- SMILES: 分子的SMILES表示
- BBB+/BBB-: 血脑屏障渗透性标签（BBB+ 或 BBB-）
- 其他特征列
## 特征提取和处理流程

1. 数据预处理
2. 使用MolTransNet提取图特征
3. 使用GEMM提取3D特征
4. 使用SmileTransformer处理SMILES字符串
5. 特征融合
6. 使用KAN模型进行最终预测

## 模型训练和评估流程

1. 数据预处理和特征提取
2. 特征融合
3. 数据增强和重采样
4. 超参数优化（使用Optuna）
5. 模型训练（使用早停策略）
6. 在测试集上评估模型性能
## 重要依赖

- Python 3.8+
- PyTorch 1.9.0
- RDKit 2023.9.6
- NumPy 1.21.0
- Pandas 1.3.0
- Scikit-learn 0.24.2
- Optuna 2.10.0
- Colorlog 6.7.0
- tqdm 4.62.3

## 注意事项

- 确保输入数据的格式正确，并包含所有必要的列。
- 在大规模数据集上训练时，可能需要调整批次大小和学习率以获得最佳性能。
- 预测结果将保存在指定的输出路径中，格式为CSV文件。

