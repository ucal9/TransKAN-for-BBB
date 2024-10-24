---

# KAN模型预测血脑屏障(BBB)渗透性

本项目使用基于Kolmogorov-Arnold表示定理并整合B样条的多模态模型来预测分子的血脑屏障(BBB)渗透性。

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

- `--data_file`: 输入数据文件路径（默认：'./data/b3db_fused_features64d.csv'）
- `--label_col`: 标签列名称（默认：'BBB+/BBB-'）
- `--augment_factor`: 数据增强因子（默认：2）
- `--resample_strategy`: 重采样策略（默认：'auto'）
- `--random_seed`: 随机种子（默认：42）
- `--num_basis`: B样条基函数数量（默认：10）
- `--degree`: B样条阶数（默认：3）

### 示例

1. 使用默认参数运行：
   ```
   python TransKAN.py
   ```

2. 使用自定义参数运行：
   ```
   python TransKAN.py --data_file my_data.csv --label_col BBB_label --num_basis 15 --degree 4
   ```

## 数据格式

输入数据应为CSV格式，包含以下列：
- SMILES: 分子的SMILES表示
- BBB+/BBB-: 血脑屏障渗透性标签（BBB+ 或 BBB-）
- 其他特征列

## 特征提取和处理

本项目使用多种方法提取和处理分子特征，以提高BBB渗透性预测的准确性：

1. **MolTransNet: 分子图神经网络与Transformer**
2. **SmileTransformer: SMILES字符串处理**
3. **GEMM: 3D特征提取**
4. **RDKit: Morgan指纹生成**
5. **特征融合: 综合特征向量生成**
6. **Attention-based Feature Fusion: 化学知识增强的注意力机制**

## KAN模型架构

KAN模型结合了Kolmogorov-Arnold表示定理和B样条理论，主要包括以下组件：

1. **B样条增强的KolmogorovLayer**: 
   - 内部函数：使用B样条基函数对每个输入特征进行非线性变换。
   - 外部函数：组合内部函数的输出。
2. **多层KolmogorovLayer**: 堆叠多个层，每层后跟ReLU激活、批量归一化和Dropout。
3. **可调节的超参数**: 包括B样条的基函数数量(num_basis)和阶数(degree)。

## 模型训练和评估流程

1. 数据预处理和增强
2. 交叉验证（10折）
3. 超参数优化（使用Optuna）
4. 训练最终模型
5. 在测试集上评估模型性能

## 输出结果

脚本运行后，结果将保存在 'result' 目录下，包括：
- training.log：训练过程的日志
- best_params.json：最佳超参数
- final_model.pt：最终训练的模型
- 每个折叠的检查点文件
- UMAP可视化结果

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
- SciPy 1.7.0 (用于B样条实现)

## 注意事项

- 确保输入数据的格式正确，并包含所有必要的列。
- 在大规模数据集上训练时，可能需要调整批次大小、学习率以及B样条参数以获得最佳性能。
- 预测结果将保存在指定的输出路径中，格式为CSV文件。
- B样条参数(num_basis和degree)对模型性能有重要影响，可能需要针对特定数据集进行调优。
