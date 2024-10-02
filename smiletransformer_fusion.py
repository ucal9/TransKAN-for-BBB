import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, matthews_corrcoef)
from tqdm import tqdm
import logging

# 设置日志系统
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 读取数据并映射标签
data_file = './data/combined data/B3DB/B3DB_combined_desc.csv'
df = pd.read_csv(data_file)
df['BBB+/BBB-'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
smiles_list = df['SMILES'].tolist()
labels = df['BBB+/BBB-'].tolist()

# 打印原始样本数量
logging.info(f"Original dataset distribution:\n{pd.Series(labels).value_counts()}")

# 特征提取函数，将SMILES转换为64维度的特征
def get_smiles_features(smiles_list, max_len=64):
    features = np.zeros((len(smiles_list), max_len), dtype=int)
    char_dict = {c: i for i, c in enumerate(set(''.join(smiles_list)))}

    for i, smiles in enumerate(smiles_list):
        for j, char in enumerate(smiles[:max_len]):
            features[i, j] = char_dict[char]

    return features

# 提取64维度的特征
features_64d = get_smiles_features(smiles_list, max_len=64)

# 打印提取的特征数量
logging.info(f"Number of features extracted: {features_64d.shape[1]}")

# 数据集划分
X_train_64d, X_temp_64d, y_train_64d, y_temp_64d = train_test_split(features_64d, labels, test_size=0.3, random_state=42)
X_val_64d, X_test_64d, y_val_64d, y_test_64d = train_test_split(X_temp_64d, y_temp_64d, test_size=0.5, random_state=42)

# 打印训练集、验证集和测试集的样本分布
logging.info(f"Training set distribution:\n{pd.Series(y_train_64d).value_counts()}")
logging.info(f"Validation set distribution:\n{pd.Series(y_val_64d).value_counts()}")
logging.info(f"Test set distribution:\n{pd.Series(y_test_64d).value_counts()}")

# 数据加载器
train_dataset = TensorDataset(torch.tensor(X_train_64d, dtype=torch.long), torch.tensor(y_train_64d, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val_64d, dtype=torch.long), torch.tensor(y_val_64d, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test_64d, dtype=torch.long), torch.tensor(y_test_64d, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 自定义 Transformer 模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes, max_len, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = self.create_positional_encoding(max_len, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def create_positional_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.batch_norm(x)
        return self.fc(x)

# 定义模型和优化器
vocab_size = len(set(''.join(smiles_list)))
d_model = 64
nhead = 8
num_encoder_layers = 3
dim_feedforward = 256
num_classes = 2
max_len = 64

model = TransformerClassifier(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes, max_len)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练模型
num_epochs = 50
best_val_loss = float('inf')
early_stopping_patience = 5
early_stopping_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for batch in tepoch:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    scheduler.step()

    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {total_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')

    # 检查是否为最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0  # 重置计数器
    else:
        early_stopping_counter += 1
        logging.info(f'Validation loss did not improve for {early_stopping_counter} epochs.')

    # 如果验证集损失连续5个回合没有改善，则停止训练
    if early_stopping_counter >= early_stopping_patience:
        logging.info("Early stopping triggered. Stopping training.")
        break

# 保存所有分子的特征向量
output_features_file = './mutiple fusion/b3db_64d_features.csv'
features_df = pd.DataFrame(features_64d, columns=[f'Feature_{i + 1}' for i in range(64)])
features_df['SMILES'] = smiles_list
features_df['Label'] = labels

# 将列的顺序调整为先 'SMILES', 'Label' 再跟随特征列
features_df = features_df[['SMILES', 'Label'] + [f'Feature_{i + 1}' for i in range(64)]]

features_df.to_csv(output_features_file, index=False)

# 打印数据集和保存文件的信息
logging.info(f'Number of samples: {features_df.shape[0]}')
logging.info(f'Number of features per sample: {features_df.shape[1] - 2}')  # 减去 'SMILES' 和 'Label' 两列
logging.info(f'nlp的特征已保存到 {output_features_file}')

# 模型评估
def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    y_pred = []
    y_true = []
    y_scores = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
            y_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    
    return total_loss / len(data_loader), y_true, y_pred, y_scores

# 评估验证集
val_loss, val_true, val_pred, val_scores = evaluate_model(model, val_loader, criterion)

# 评估测试集
test_loss, test_true, test_pred, test_scores = evaluate_model(model, test_loader, criterion)

# 计算并打印评估指标
def print_metrics(y_true, y_pred, y_scores, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    mcc = matthews_corrcoef(y_true, y_pred)

    TN, FP, FN, TP = cm.ravel()
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    gmeans = math.sqrt(recall * specificity)

    print(f'\n{dataset_name} Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (PRE): {precision:.4f}')
    print(f'Recall (SE): {recall:.4f}')
    print(f'Specificity (SP): {specificity:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Gmeans: {gmeans:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'MCC: {mcc:.4f}')
    print('Confusion Matrix:')
    print(cm)

    logging.info(f'\n{dataset_name} Metrics:')
    logging.info(f'Accuracy: {accuracy:.4f}')
    logging.info(f'Precision (PRE): {precision:.4f}')
    logging.info(f'Recall (SE): {recall:.4f}')
    logging.info(f'Specificity (SP): {specificity:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    logging.info(f'Gmeans: {gmeans:.4f}')
    logging.info(f'AUC: {auc:.4f}')
    logging.info(f'MCC: {mcc:.4f}')
    logging.info(f'Confusion Matrix:\n{cm}')

# 打印验证集和测试集的评估指标
print_metrics(val_true, val_pred, val_scores, "Validation")
print_metrics(test_true, test_pred, test_scores, "Test")

