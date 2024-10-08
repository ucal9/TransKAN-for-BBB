---

# KAN模型预测血脑屏障(BBB)渗透性

本项目使用知识感知网络(Knowledge-Aware Network, KAN)模型来预测分子的血脑屏障(BBB)渗透性。

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

## 特征提取和处理

本项目使用多种方法提取和处理分子特征，以提高BBB渗透性预测的准确性：

1. **MolTransNet（分子图神经网络）**
   - 用于提取图像特征
   - 确保进行适当的图像预处理
   - 输入格式需符合模型要求

2. **SmileTransformer**
   - 处理SMILES字符串
   - 提取化学特征
   - 确保输入格式正确

3. **GEMM**
   - 提取3D特征
   - 确保数据格式符合GEMM要求
   - 进行必要的预处理

4. **RDKit**
   - 生成Morgan指纹作为额外特征

5. **特征融合**
   - 将从MolTransNet、SmileTransformer、GEMM和RDKit提取的特征拼接成一个综合特征向量

6. **KAN输入**
   - 将融合后的特征向量输入KAN模型进行训练或预测

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

