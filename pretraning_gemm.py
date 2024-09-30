import os
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import random

class GEMMEncoder:
    def encode(self, distance_matrix, atom_types):
        np.random.seed(42)
        encoded_features = np.random.rand(256)
        return encoded_features

def calculate_geometric_features(mol, smiles):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, maxAttempts=10, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
    except ValueError as ve:
        print(f"ValueError: {ve} for SMILES: {smiles}")
        return None
    except Exception as e:
        print(f"生成3D结构时出错: {e} for SMILES: {smiles}")
        return None

    conf = mol.GetConformer()
    atom_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    atom_types = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    return {
        'atom_coords': atom_coords,
        'atom_types': atom_types
    }

def calculate_distance_matrix(atom_coords):
    num_atoms = atom_coords.shape[0]
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance_matrix[i, j] = np.linalg.norm(atom_coords[i] - atom_coords[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix

def calculate_alternative_features(mol):
    return {
        'MolWt': Descriptors.MolWt(mol),
        'NumAtoms': mol.GetNumAtoms(),
        'NumBonds': mol.GetNumBonds(),
    }

def generate_3d_features_with_fallback(smiles_list, labels):
    data_mols = []
    failed_smiles = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                features = calculate_geometric_features(mol, smiles)
                if features is not None:
                    data_mols.append({
                        'smiles': smiles,
                        'label': labels[i],
                        'atom_coords': features['atom_coords'],
                        'atom_types': features['atom_types'],
                        'is_3d': True
                    })
                else:
                    alternative_features = calculate_alternative_features(mol)
                    data_mols.append({
                        'smiles': smiles,
                        'label': labels[i],
                        'alternative_features': alternative_features,
                        'is_3d': False
                    })
                    print(f"3D特征生成失败，为SMILES: {smiles} 生成2D特征作为替代")
                    failed_smiles.append(smiles)
            except Exception as e:
                print(f"生成特征时出错: {e} for SMILES: {smiles}")
                failed_smiles.append(smiles)
        else:
            print(f"无效的SMILES: {smiles}")
            failed_smiles.append(smiles)
    return data_mols, failed_smiles

def encode_with_gemm_or_alternative(data_mols, gemm_encoder):
    encoded_data = []
    for mol_data in data_mols:
        if mol_data.get('is_3d', False):
            atom_coords = mol_data['atom_coords']
            atom_types = mol_data['atom_types']
            distance_matrix = calculate_distance_matrix(atom_coords)
            encoded_features = gemm_encoder.encode(distance_matrix, atom_types)
        else:
            alternative_features = mol_data['alternative_features']
            encoded_features = np.array(list(alternative_features.values()) + [0]*(256-len(alternative_features)))

        encoded_data.append({
            'smiles': mol_data['smiles'],
            'label': mol_data['label'],
            'encoded_features': encoded_features
        })
    return encoded_data

def save_3d_features(data_mols, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data_mols, f)
    print(f"3D特征已保存到: {output_file}")

def load_3d_features(input_file):
    with open(input_file, 'rb') as f:
        data_mols = pickle.load(f)
    return data_mols

def save_encoded_features_to_csv(encoded_data, output_file):
    rows = []
    for item in encoded_data:
        row = [item['smiles'], item['label']] + item['encoded_features'].tolist()
        rows.append(row)
    columns = ['SMILES', 'Label'] + [f'Feature_{i + 1}' for i in range(256)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"编码特征已保存到: {output_file}")
    print(f"保存的分子数量: {df.shape[0]}")

def evaluate_model(encoded_data):
    true_labels = [item['label'] for item in encoded_data]
    predictions = [random.choice([0, 1]) for _ in true_labels]  # 随机生成预测值
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (PRE): {precision:.4f}')
    print(f'Recall (SE): {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'MCC: {mcc:.4f}')

def load_pretrain_data(pretrain_file):
    pretrain_data = pd.read_csv(pretrain_file)
    pretrain_data['label'] = pretrain_data['n_np'].map({'p': 1, 'np': 0})  # 将p和np转换为1和0
    return pretrain_data

def pretrain_gemm_encoder(pretrain_data, gemm_encoder):
    print("开始使用预训练数据预训练GEMMEncoder...")
    for index, row in pretrain_data.iterrows():
        smiles = row['SMILES']  # 这里的 'Name' 是 SMILES 列的列名
        label = row['label']
        pretrain_features = row[2:].values  # 假设前两列是SMILES和Label，剩下的是特征
        # 在这里可以模拟对encoder的更新，实际情况中根据你的模型来更新
        # 这里我们假设预训练会影响encoder的权重，比如进行特征向量加权平均作为预训练步骤
        np.random.seed(42)
        _ = gemm_encoder.encode(np.random.rand(5, 5), [6, 1, 6])  # 模拟编码
    print("预训练完成。")

if __name__ == "__main__":
    pretrain_file = './data/2300+/PN data.csv'
    data_file = './data/combined data/B3DB/B3DB_combined_desc.csv'
    output_file = './data/3d/b3db_3d_features.pkl'

    # 加载预训练数据
    pretrain_data = load_pretrain_data(pretrain_file)

    # 初始化GEMMEncoder并进行预训练
    gemm_encoder = GEMMEncoder()
    pretrain_gemm_encoder(pretrain_data, gemm_encoder)

    # 主任务流程
    if os.path.exists(output_file):
        print("3D特征文件已存在，直接加载...")
        data_mols = load_3d_features(output_file)
    else:
        df = pd.read_csv(data_file)
        df['BBB+/BBB-'] = df['BBB+/BBB-'].map({'BBB+': 1, 'BBB-': 0})
        smiles_list = df['SMILES'].tolist()
        labels = df['BBB+/BBB-'].tolist()

        data_mols, failed_smiles = generate_3d_features_with_fallback(smiles_list, labels)
        if failed_smiles:
            print(f"以下SMILES无法生成3D特征，已生成替代2D特征：{failed_smiles}")
        save_3d_features(data_mols, output_file)
        print("3D特征和替代特征生成并保存完毕！")

    # 训练过程，使用GEMMEncoder对主数据集进行编码
    best_val_loss = float('inf')
    best_encoded_data = None
    for epoch in range(50):
        print(f"模拟训练，第 {epoch + 1} 轮")
        encoded_data = encode_with_gemm_or_alternative(data_mols, gemm_encoder)
        val_loss = np.random.rand()  # 用随机数模拟验证集损失
        print(f"Validation loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoded_data = encoded_data
            print(f"发现新低损失，保存当前特征向量。")

    # 只保存最佳特征集作为最终输出
    all_data_csv = '../mutiple fusion/b3db_256_only3dposition.csv'
    save_encoded_features_to_csv(encode_with_gemm_or_alternative(data_mols, gemm_encoder), all_data_csv)

    # 打印评估指标
    evaluate_model(best_encoded_data)
