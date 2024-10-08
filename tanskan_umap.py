import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
from tqdm import tqdm
import optuna
import colorlog
import json
from sklearn.metrics import confusion_matrix
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 设置日志
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='result/checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path  # 保存模型检查点的路径

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型 ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class KAN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=3, dropout=0.1, random_state=None):
        super(KAN, self).__init__()
        self.random_state = random_state
        torch.manual_seed(random_state)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))  # 添加BatchNorm
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))  # 添加BatchNorm
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def fit(self, X, y, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001, early_stopping=None):
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(pd.get_dummies(y)['BBB+'].values).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(pd.get_dummies(y_val)['BBB+'].values).unsqueeze(1)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        best_val_auc = 0
        best_epoch = 0
        patience = 10
        no_improve = 0

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)

            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            # Calculate metrics
            train_metrics = self.calculate_metrics(X_tensor, y_tensor)
            val_metrics = self.calculate_metrics(X_val_tensor, y_val_tensor)

            scheduler.step(val_loss)
            if early_stopping:
                early_stopping(val_loss, self)

            logger.info(f'Epoch [{epoch + 1}/{epochs}]:')
            logger.info(f'  Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
            logger.info(f'  Train Metrics: {train_metrics}')
            logger.info(f'  Val Metrics: {val_metrics}')

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch + 1
                torch.save(self.state_dict(), 'best_model.pt')
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f"Early stopping triggered. No improvement for {patience} epochs.")
                break

            if early_stopping and early_stopping.early_stop:
                logger.info(f"Early stopping triggered. Best epoch: {best_epoch}")
                break

        logger.info(f"Best epoch: {best_epoch} with validation AUC: {best_val_auc:.4f}")
        self.load_state_dict(torch.load('best_model.pt'))

    def calculate_metrics(self, X, y):
        with torch.no_grad():
            outputs = self(X)
            y_pred = (outputs > 0.5).float()
            y_pred_proba = outputs.numpy()

        y_true = y.numpy()
        y_pred = y_pred.numpy()

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            outputs = self(X_tensor)
            predictions = (outputs > 0.5).float().numpy().flatten()
        return np.where(predictions == 1, 'BBB+', 'BBB-')

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            outputs = self(X_tensor).numpy()
        return np.column_stack((1 - outputs, outputs))


def augment_smiles(smiles, num_augments):
    return [smiles] * num_augments


def preprocess_and_augment_bbb_data(data_filename, smile_col_name='SMILE', label_col_name='BBB+/BBB-', augment_factor=2,
                                    feature_cols=None):
    data = pd.read_csv(data_filename)
    Y = data[label_col_name]
    SMILES = data[smile_col_name]

    if feature_cols is None:
        X = data.drop(columns=[smile_col_name, label_col_name])
    else:
        X = data[feature_cols]

    logger.info("原始特征：")
    logger.info(X.head())

    # 打印原始数据的BBB+和BBB-的分布
    logger.info("原始数据的BBB+/BBB-分布：")
    logger.info(Y.value_counts())

    augmented_X, augmented_Y, augmented_SMILES = [], [], []
    for smiles, x, y in tqdm(zip(SMILES, X.values, Y), total=len(SMILES), desc="Augmenting data"):
        augmented_SMILES.append(smiles)
        augmented_X.append(x)
        augmented_Y.append(y)

        augmented_smiles = augment_smiles(smiles, num_augments=augment_factor)
        for aug_smiles in augmented_smiles:
            augmented_SMILES.append(aug_smiles)
            augmented_X.append(x)
            augmented_Y.append(y)

    augmented_X = pd.DataFrame(augmented_X, columns=X.columns)
    augmented_Y = pd.Series(augmented_Y)

    logger.info(f"增强后的数据量: {len(augmented_X)}")

    # 打印增强后数据的BBB+和BBB-的分布
    logger.info("增强后据的BBB+/BBB-分布：")
    logger.info(augmented_Y.value_counts())

    return augmented_X, augmented_Y, augmented_SMILES


def resample(X, y, strategy='auto', random_state=None):
    smote = SMOTE(sampling_strategy=strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logger.info(f"重采样后的数据量: {len(X_resampled)}")

    # 打印重采样后数据的BBB+和BBB-的分布
    logger.info("重采样���数据的BBB+/BBB-分布：")
    logger.info(pd.Series(y_resampled).value_counts())

    return X_resampled, y_resampled


def objective(trial, X_train, y_train, X_val, y_val):
    config = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-1),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_categorical("epochs", [10, 30, 50]),
        "random_seed": trial.suggest_int("random_seed", 0, 100000)
    }

    model = KAN(
        input_dim=X_train.shape[1],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        random_state=config["random_seed"]
    )
    model.fit(X_train, y_train, X_val, y_val, epochs=config["epochs"], batch_size=config["batch_size"], learning_rate=config["lr"])

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy


def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # 保存最佳参数到文件
    os.makedirs('result', exist_ok=True)  # 创建result目录（如果不存在）
    with open('result/best_params.json', 'w') as f:
        json.dump(trial.params, f)

    return trial.params


def load_best_params():
    if os.path.exists('result/best_params.json'):
        with open('result/best_params.json', 'r') as f:
            return json.load(f)
    return None


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    tn, fp, fn, tp = confusion_matrix(y == 'BBB+', y_pred == 'BBB+').ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    gmeans = np.sqrt(sensitivity * specificity)
    auc = roc_auc_score(y == 'BBB+', y_pred_proba)
    
    return {
        'ACC': accuracy,
        'PRE': precision,
        'SE': sensitivity,
        'SP': specificity,
        'Gmeans': gmeans,
        'F': f1,
        'AUC': auc
    }


def umap_visualization(X, y, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(X)
    
    # 将标签转换为数值
    y_numeric = (y == 'BBB+').astype(int)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_numeric, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('UMAP projection of the B3DB')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.savefig('result/transkan_umap.png')
    plt.close()

def shap_analysis(model, X):
    background = shap.DeepExplainer(model, torch.FloatTensor(X.values[:100]))
    shap_values = background.shap_values(torch.FloatTensor(X.values))

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('result/shap_summary.png')
    plt.close()

    plt.figure(figsize=(20, 12))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig('result/shap_summary_dot.png')
    plt.close()

def main():
    # 设置参数
    data_file = './mutiple fusion/b3db_fused_features64d.csv'
    label_col = 'BBB+/BBB-'
    augment_factor = 2
    resample_strategy = 'auto'
    random_seed = 42

    X, y, SMILES = preprocess_and_augment_bbb_data(data_file,
                                                   smile_col_name='SMILES',
                                                   label_col_name=label_col,
                                                   augment_factor=augment_factor)

    # 首先分割出测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)
    
    # 如果result目录不存在，则创建它
    os.makedirs('result', exist_ok=True)

    # 设置日志记录到文件
    file_handler = logging.FileHandler('result/training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    fold_results = []
    for fold, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val), 1):
        logger.info(f"Fold {fold}")
        
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        X_resampled, y_resampled = resample(X_train, y_train, strategy=resample_strategy,
                                            random_state=random_seed)

        best_params = load_best_params()
        if best_params is None:
            best_params = optimize_hyperparameters(X_resampled, y_resampled, X_val, y_val, n_trials=100)
        else:
            logger.info("使用之前保存的最佳参数")

        # 减少epochs数量
        best_params["epochs"] = 50  # 减少到50个epoch

        model = KAN(
            input_dim=X_resampled.shape[1],
            hidden_dim=best_params["hidden_dim"],
            num_layers=best_params["num_layers"],
            dropout=best_params["dropout"],
            random_state=best_params["random_seed"]
        )
        early_stopping = EarlyStopping(patience=10, verbose=True, path=f'result/checkpoint_fold_{fold}.pt')
        model.fit(X_resampled, y_resampled, X_val, y_val, epochs=best_params["epochs"], 
                  batch_size=best_params["batch_size"], learning_rate=best_params["lr"],
                  early_stopping=early_stopping)

        metrics = evaluate_model(model, X_val, y_val)
        fold_results.append(metrics)

        logger.info(f"Fold {fold} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric:6s}: {value:.3f}")

    # Calculate and print average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
    std_metrics = {metric: np.std([fold[metric] for fold in fold_results]) for metric in fold_results[0]}

    logger.info("Cross-validation Results:")
    for metric in avg_metrics:
        logger.info(f"  {metric:6s}: {avg_metrics[metric]:.3f} (+/- {std_metrics[metric]:.3f})")

    # Train final model on all training data
    X_resampled, y_resampled = resample(X_train_val, y_train_val, strategy=resample_strategy, random_state=random_seed)
    final_model = KAN(
        input_dim=X_resampled.shape[1],
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        random_state=best_params["random_seed"]
    )
    final_model.fit(X_resampled, y_resampled, X_resampled, y_resampled, epochs=best_params["epochs"], 
                    batch_size=best_params["batch_size"], learning_rate=best_params["lr"])

    # Evaluate final model on test set
    test_metrics = evaluate_model(final_model, X_test, y_test)
    logger.info("Test Set Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric:6s}: {value:.3f}")

    # Save the final model
    torch.save(final_model.state_dict(), 'result/final_model.pt')
    logger.info("最终模型已保存到 'result/final_model.pt'")

    # SHAP analysis
    shap_analysis(final_model, X_test)

    # UMAP visualization for test set
    umap_visualization(X_test, y_test)

if __name__ == "__main__":
    main()