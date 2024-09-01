import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config
from data_process import feature_filter, augment_smiles, scale_data
from utils import DataLogger
import resampling

log = DataLogger(__name__).getlog()

def get_X_Y_argument(data_filename, label, smile_col_name='SMILES', label_col_name='BBB+/BBB-', filter_feature=True, augment_factor=1):
    # 读取数据
    data = pd.read_csv(data_filename)
    Y = data[label_col_name]
    SMILES = data[smile_col_name]
    X = data.drop(columns=[smile_col_name, label_col_name])

    # 可选：筛选特征
    if filter_feature:
        X = feature_filter(X)

    # 数据增强：翻倍数据集
    augmented_X = []
    augmented_Y = []
    augmented_SMILES = []

    for smiles, x, y in zip(SMILES, X.values, Y):
        # 增加原始数据
        augmented_SMILES.append(smiles)
        augmented_X.append(x)
        augmented_Y.append(y)

        # 增强后的数据
        augmented_smiles = augment_smiles(smiles, num_augments=augment_factor)
        for aug_smiles in augmented_smiles:
            augmented_SMILES.append(aug_smiles)
            augmented_X.append(x)  # 使用相同的特征
            augmented_Y.append(y)  # 使用相同的标签

    # 转换为DataFrame
    augmented_X = pd.DataFrame(augmented_X, columns=X.columns)
    augmented_Y = pd.Series(augmented_Y)
    augmented_SMILES = pd.Series(augmented_SMILES)

    return augmented_X, augmented_Y.ravel(), augmented_SMILES

def plot_sample_distribution(y):
    # 绘制正负样本分布的柱状图，并添加数量显示
    sns.countplot(x=y)
    plt.title('Distribution of BBB+/BBB- Samples')
    plt.xlabel('Label')
    plt.ylabel('Count')

    # 在每个柱上显示数量
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.show()

def main():
    # 获取并预处理数据
    X, y, SMILES = get_X_Y_argument('./mutiple fusion/b3db_fused_features64d.csv', label=['BBB+', 'BBB-'],
                                    smile_col_name='SMILES', label_col_name='BBB+/BBB-', filter_feature=True)

    # 数据标准化
    X = scale_data(X, "MinMax")

    # 进行重采样
    X_resampled, y_resampled = resampling.resample(X, y, config.resample_strategy, random_state=config.random_seed)

    # 绘制重采样后的正负样本分布
    plot_sample_distribution(y_resampled)

if __name__ == "__main__":
    main()
