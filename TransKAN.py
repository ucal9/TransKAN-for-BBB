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

# 设置日志
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
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
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

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
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
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

    def fit(self, X, y, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(pd.get_dummies(y)['BBB+'].values).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(pd.get_dummies(y_val)['BBB+'].values).unsqueeze(1)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=10, verbose=True)

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

            if early_stopping.early_stop:
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
    logger.info("增强后数据的BBB+/BBB-分布：")
    logger.info(augmented_Y.value_counts())

    return augmented_X, augmented_Y, augmented_SMILES


def resample(X, y, strategy='auto', random_state=None):
    smote = SMOTE(sampling_strategy=strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logger.info(f"重采样后的数据量: {len(X_resampled)}")

    # 打印重采样后数据的BBB+和BBB-的分布
    logger.info("重采样后数据的BBB+/BBB-分布：")
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
    with open('best_params.json', 'w') as f:
        json.dump(trial.params, f)

    return trial.params


def load_best_params():
    if os.path.exists('best_params.json'):
        with open('best_params.json', 'r') as f:
            return json.load(f)
    return None


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label='BBB+')
    recall = recall_score(y, y_pred, pos_label='BBB+')
    f1 = f1_score(y, y_pred, pos_label='BBB+')
    auc = roc_auc_score(y == 'BBB+', y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def main(args):
    X, y, SMILES = preprocess_and_augment_bbb_data(args.data_file,
                                                   smile_col_name=args.smile_col,
                                                   label_col_name=args.label_col,
                                                   augment_factor=args.augment_factor)

    # 首先分割出测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_seed, stratify=y)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.random_seed)
    
    fold_results = []
    for fold, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val), 1):
        logger.info(f"Fold {fold}")
        
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        X_resampled, y_resampled = resample(X_train, y_train, strategy=args.resample_strategy,
                                            random_state=args.random_seed)

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
        model.fit(X_resampled, y_resampled, X_val, y_val, epochs=best_params["epochs"], 
                  batch_size=best_params["batch_size"], learning_rate=best_params["lr"])

        metrics = evaluate_model(model, X_val, y_val)
        fold_results.append(metrics)

        logger.info(f"Fold {fold} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.capitalize()}:  {value:.4f}")

    # Calculate and print average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
    std_metrics = {metric: np.std([fold[metric] for fold in fold_results]) for metric in fold_results[0]}

    logger.info("Cross-validation Results:")
    for metric in avg_metrics:
        logger.info(f"  {metric.capitalize()}:  {avg_metrics[metric]:.4f} (+/- {std_metrics[metric]:.4f})")

    # Train final model on all training data
    X_resampled, y_resampled = resample(X_train_val, y_train_val, strategy=args.resample_strategy, random_state=args.random_seed)
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
        logger.info(f"  {metric.capitalize()}:  {value:.4f}")

    # Save the final model
    torch.save(final_model.state_dict(), 'final_model.pt')
    logger.info("Final model saved to 'final_model.pt'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='KAN model for BBB prediction with hyperparameter optimization')
    parser.add_argument('--data_file', type=str, default='./mutiple fusion/b3db_fused_features64d.csv',
                        help='Path to the input data file')
    parser.add_argument('--smile_col', type=str, default='SMILES', help='Name of the SMILES column')
    parser.add_argument('--label_col', type=str, default='BBB+/BBB-', help='Name of the label column')
    parser.add_argument('--augment_factor', type=int, default=2, help='Data augmentation factor')
    parser.add_argument('--resample_strategy', type=str, default='auto', help='Resampling strategy')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    main(args)
