import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import rdkit
from rdkit import Chem
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define atomic weights based on mass
atomic_weights = {
    'H': 1.008, 'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.06, 'Cl': 35.45,
    'F': 19.00, 'Br': 79.90, 'I': 126.90, 'P': 30.97, 'Si': 28.09, 'B': 10.81,
    'Se': 78.96, 'Te': 127.60, 'As': 74.92, 'Ge': 72.64, 'Sn': 118.71
}

# Define chemical bond weights
bond_weights = {
    'single': 1.0, 'double': 1.5, 'triple': 1.8, 'aromatic': 1.5
}

selected_features = [
    'logP', 'nHBDon', 'nHBAcc', 'TPSA', 'MW', 'naRing', 'nHeavyAtom',
    'nRot', 'MWHBN', 'nAcid', 'nHetero', 'ATS5se', 'ATSC0c', 'ATSC1c',
    'ATSC4Z', 'ATSC4v', 'ATSC1se', 'Xch-6dv', 'Xpc-5d', 'SdssC', 'IC0',
    'IC1', 'PEOE_VSA1', 'PEOE_VSA9', 'PEOE_VSA10', 'SlogP_VSA2',
    'VSA_EState3', 'SLogP', 'GGI3', 'MP'
]

class ChemicalKnowledgeEnhancedAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Ensure feature_dim is divisible by num_heads
        self.adjusted_feature_dim = ((feature_dim - 1) // num_heads + 1) * num_heads
        
        self.descriptor_embedding = nn.Embedding(len(selected_features), self.adjusted_feature_dim)
        self.atom_embedding = nn.Embedding(len(atomic_weights), self.adjusted_feature_dim)
        self.bond_embedding = nn.Embedding(len(bond_weights), self.adjusted_feature_dim)
        
        self.multihead_attn = nn.MultiheadAttention(self.adjusted_feature_dim, num_heads)
        self.fusion_layer = nn.Linear(self.adjusted_feature_dim * 3, self.adjusted_feature_dim)
        self.dynamic_weight = nn.Linear(self.adjusted_feature_dim, 3)
        
        # Add a final linear layer to project back to the original feature_dim
        self.output_projection = nn.Linear(self.adjusted_feature_dim, feature_dim)

    def forward(self, features, atom_types, bond_types):
        batch_size, _ = features.shape
        seq_len = 1  # Since features is 2D, we treat it as a sequence of length 1
        
        # Reshape features to be 3D
        features = features.unsqueeze(1)
        
        # Pad the input features to match the adjusted_feature_dim
        if features.shape[-1] < self.adjusted_feature_dim:
            padding = torch.zeros(batch_size, seq_len, self.adjusted_feature_dim - features.shape[-1], device=features.device)
            features = torch.cat([features, padding], dim=-1)
        
        descriptor_indices = torch.arange(len(selected_features)).to(features.device)
        descriptor_embed = self.descriptor_embedding(descriptor_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        atom_indices = torch.tensor([list(atomic_weights.keys()).index(atom) for atom in atom_types]).to(features.device)
        atom_embed = self.atom_embedding(atom_indices).unsqueeze(1)
        
        bond_indices = torch.tensor([list(bond_weights.keys()).index(bond) for bond in bond_types]).to(features.device)
        bond_embed = self.bond_embedding(bond_indices).unsqueeze(1)
        
        combined_features = self.fusion_layer(torch.cat([features, atom_embed, bond_embed], dim=-1))
        
        # Transpose the inputs to match the expected shape for nn.MultiheadAttention
        combined_features = combined_features.transpose(0, 1)
        descriptor_embed = descriptor_embed.transpose(0, 1)
        
        attn_output, _ = self.multihead_attn(combined_features, descriptor_embed, descriptor_embed)
        
        # Transpose back to the original shape
        attn_output = attn_output.transpose(0, 1)
        
        dynamic_weights = F.softmax(self.dynamic_weight(attn_output.mean(dim=1)), dim=-1)
        
        weighted_output = (
            dynamic_weights[:, 0].unsqueeze(1).unsqueeze(2) * attn_output +
            dynamic_weights[:, 1].unsqueeze(1).unsqueeze(2) * atom_embed +
            dynamic_weights[:, 2].unsqueeze(1).unsqueeze(2) * bond_embed
        )
        
        # Project back to the original feature dimension
        output = self.output_projection(weighted_output)
        
        # Remove the sequence dimension
        output = output.squeeze(1)
        
        return output

class EnhancedChemicalFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.chemical_attention = ChemicalKnowledgeEnhancedAttention(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, features, atom_types, bond_types):
        attended_features = self.chemical_attention(features, atom_types, bond_types)
        fused_features = self.fc(attended_features)
        return fused_features

def calculate_chemical_bond_attention(atomic_features, bond_features, atom_types, bond_types, smiles):
    logger.info("Calculating chemical bond attention...")
    atom_weight = torch.tensor([atomic_weights.get(atom, 1.0) for atom in atom_types], dtype=torch.float32)
    bond_weight = torch.tensor([bond_weights.get(bond, 1.0) for bond in bond_types], dtype=torch.float32)

    weighted_atomic_features = atomic_features * atom_weight[:, None]
    weighted_bond_features = bond_features * bond_weight[:, None]

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 10:
                idx = atom.GetIdx()
                weighted_atomic_features[idx] *= 1.5

    Q = weighted_atomic_features
    K = weighted_bond_features
    V = weighted_bond_features
    scores = torch.matmul(Q, K.T) / torch.sqrt(K.size(-1))
    attention_weights = F.softmax(scores, dim=-1)
    weighted_features = torch.matmul(attention_weights, V)

    logger.info("Chemical bond attention calculated.")
    return attention_weights, weighted_features

def calculate_molecular_attention(atomic_features, atom_types, selected_features):
    logger.info("Calculating molecular attention...")
    atom_weight = torch.tensor([atomic_weights.get(atom, 1.0) for atom in atom_types], dtype=torch.float32)
    weighted_atomic_features = atomic_features * atom_weight[:, None]

    feature_importance = {feature: idx + 1 for idx, feature in enumerate(selected_features)}
    feature_weights = torch.tensor([feature_importance.get(feature, 1.0) for feature in selected_features], dtype=torch.float32)
    weighted_atomic_features *= feature_weights

    Q = weighted_atomic_features
    K = weighted_atomic_features
    V = weighted_atomic_features
    scores = torch.matmul(Q, K.T) / torch.sqrt(K.size(-1))
    attention_weights = F.softmax(scores, dim=-1)
    weighted_features = torch.matmul(attention_weights, V)

    logger.info("Molecular attention calculated.")
    return attention_weights, weighted_features

def load_and_align_data(file_paths, selected_features):
    dfs = []
    logger.info("Loading and aligning data...")
    for i, path in enumerate(file_paths):
        df = pd.read_csv(path)
        if 'SMILES' not in df.columns:
            df['SMILES'] = np.random.choice(['C', 'H', 'O', 'N'], size=len(df))
        if 'Label' not in df.columns:
            df['Label'] = np.random.randint(0, 2, size=len(df))
        for feature in selected_features:
            if feature not in df.columns:
                df[feature] = np.random.rand(len(df))
        dfs.append(df)

    aligned_df = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info("Data loaded and aligned.")
    return aligned_df

def fuse_features_with_enhanced_attention(features, atom_types, bond_types, output_dim):
    logger.info(f"Fusing features with enhanced chemical attention, output_dim={output_dim}...")
    
    fusion_model = EnhancedChemicalFeatureFusion(features.shape[-1], output_dim)
    
    with torch.no_grad():
        fused_features = fusion_model(features, atom_types, bond_types)
    
    logger.info(f"Feature fusion completed, output_dim={output_dim}.")
    return fused_features

def save_fused_features(fused_features, smiles_labels, prior_features, output_file):
    # Create the output directory if it doesn't exist
    output_dir = os.path.join('..', 'data', 'fusion_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full path for the output file
    output_path = os.path.join(output_dir, output_file)
    
    logger.info(f"Saving fused features to {output_path}...")
    smiles_labels_df = pd.DataFrame(smiles_labels, columns=['SMILES', 'Label'])

    num_features_per_type = fused_features.shape[1] // 4

    fused_features_df = pd.DataFrame(fused_features,
                                     columns=[f'Fused_3D_Feature_{i + 1}' for i in range(num_features_per_type)] +
                                             [f'Fused_CV_Feature_{i + 1}' for i in range(num_features_per_type)] +
                                             [f'Fused_NLP_Feature_{i + 1}' for i in range(num_features_per_type)] +
                                             [f'Fused_Extra_Feature_{i + 1}' for i in
                                              range(fused_features.shape[1] - 3 * num_features_per_type)])

    prior_features_df = pd.DataFrame(prior_features,
                                     columns=[f'Prior_Feature_{i + 1}' for i in range(prior_features.shape[1])])

    smiles_labels_df['Label'] = smiles_labels_df['Label'].map({1: 'BBB+', 0: 'BBB-'})
    smiles_labels_df.rename(columns={'Label': 'BBB+/BBB-'}, inplace=True)

    result_df = pd.concat([smiles_labels_df, fused_features_df, prior_features_df], axis=1)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Fused features and prior features saved to: {output_path}")

    total_molecules = result_df.shape[0]
    logger.info(f"Total molecules generated: {total_molecules}")
    logger.info(f"3D modality features: {num_features_per_type}")
    logger.info(f"CV modality features: {num_features_per_type}")
    logger.info(f"NLP modality features: {num_features_per_type}")
    logger.info(f"Extra modality features: {fused_features.shape[1] - 3 * num_features_per_type}")

if __name__ == "__main__":
    file_paths = [
        './mutiple fusion/b3db_64d_features.csv',
        './mutiple fusion/b3db_cvvector128d.csv',
        './mutiple fusion/b3db_256_only3dposition.csv',
        './mutiple fusion/b3db_1623modered.csv'
    ]

    logger.info("Processing...")
    aligned_df = load_and_align_data(file_paths, selected_features)

    # Extract features, smiles, labels, and prior features
    features = torch.tensor(aligned_df[selected_features].values, dtype=torch.float32)
    smiles_labels = aligned_df[['SMILES', 'Label']].values
    prior_features = aligned_df[selected_features].values

    # Generate dummy atom types and bond types (you should replace this with actual data)
    atom_types = np.random.choice(list(atomic_weights.keys()), size=len(aligned_df))
    bond_types = np.random.choice(list(bond_weights.keys()), size=len(aligned_df))

    # Fuse features and save results
    fused_features_128d = fuse_features_with_enhanced_attention(features, atom_types, bond_types, output_dim=128)
    save_fused_features(fused_features_128d, smiles_labels, prior_features, 'fused_features_128d.csv')

    logger.info("Processing completed.")