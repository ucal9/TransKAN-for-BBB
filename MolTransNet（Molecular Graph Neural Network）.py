import timeit
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
import logging
from logging.handlers import RotatingFileHandler
import os
from collections import defaultdict
from rdkit import Chem

# 设置日志记录
def setup_logger(log_file='training.log', max_bytes=10*1024*1024, backup_count=5):
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

log = setup_logger()

# Preprocess functions
def create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(list)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        return np.array([fingerprint_dict[a] for a in atoms])

    nodes = atoms
    i_jedge_dict = i_jbond_dict

    for _ in range(radius):
        nodes_ = []
        for i, j_edge in i_jedge_dict.items():
            neighbors = [(nodes[j], edge) for j, edge in j_edge]
            fingerprint = (nodes[i], tuple(sorted(neighbors)))
            nodes_.append(fingerprint_dict[fingerprint])

        i_jedge_dict_ = defaultdict(list)
        for i, j_edge in i_jedge_dict.items():
            for j, edge in j_edge:
                both_side = tuple(sorted((nodes[i], nodes[j])))
                edge = edge_dict[(both_side, edge)]
                i_jedge_dict_[i].append((j, edge))

        nodes = nodes_
        i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def split_dataset(dataset, ratio):
    np.random.seed(1234)
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]

def create_datasets(task, dataset, radius, device):
    dir_dataset = f'./dataset/{task}/{dataset}/'
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filename):
        log.info(f"Processing {filename} for {task} task")
        with open(dir_dataset + filename, 'r') as f:
            smiles_property = f.readline().strip().split()
            data_original = f.read().strip().split('\n')

        data_original = [data for data in data_original if '.' not in data.split()[0]]
        dataset = []

        for data in data_original:
            smiles, property = data.strip().split()
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)

            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            if task == 'classification':
                property = torch.LongTensor([int(property)]).to(device)
            elif task == 'regression':
                property = torch.FloatTensor([[float(property)]]).to(device)

            dataset.append((fingerprints, adjacency, molecular_size, property, smiles))

        return dataset

    dataset_train = create_dataset('data_train.txt')
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = create_dataset('data_test.txt')

    return dataset_train, dataset_dev, dataset_test, len(fingerprint_dict)

# Model classes
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        return self.transformer(x).squeeze(1)

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output, task, nhead=4, num_transformer_layers=2):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
        self.fc = nn.Linear(dim, 64)
        self.transformer = TransformerEncoder(64, nhead, num_transformer_layers)
        self.W_output = nn.ModuleList([nn.Linear(64, 64) for _ in range(layer_output)])
        self.task = task
        if task == 'classification':
            self.W_property = nn.Linear(64, 2)
        elif task == 'regression':
            self.W_property = nn.Linear(64, 1)

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(next(self.parameters()).device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        return torch.stack([torch.sum(v, 0) for v in torch.split(vectors, axis)])

    def mean(self, vectors, axis):
        return torch.stack([torch.mean(v, 0) for v in torch.split(vectors, axis)])

    def gnn(self, inputs):
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(len(self.W_fingerprint)):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)

        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        return molecular_vectors

    def mlp(self, vectors):
        for layer in self.W_output:
            vectors = torch.relu(layer(vectors))
        return self.W_property(vectors)

    def get_molecular_vectors(self, inputs):
        molecular_vectors = self.gnn(inputs)
        molecular_vectors = self.fc(molecular_vectors)
        return self.transformer(molecular_vectors)

    def forward_classifier(self, data_batch, train):
        inputs = data_batch[:-1]
        correct_labels = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.get_molecular_vectors(inputs)
            predicted_scores = self.mlp(molecular_vectors)
            return F.cross_entropy(predicted_scores, correct_labels)
        else:
            with torch.no_grad():
                molecular_vectors = self.get_molecular_vectors(inputs)
                predicted_scores = self.mlp(molecular_vectors)
            predicted_scores = predicted_scores.to('cpu').data.numpy()
            predicted_scores = [s[1] for s in predicted_scores]
            correct_labels = correct_labels.to('cpu').data.numpy()
            return predicted_scores, correct_labels

    def forward_regressor(self, data_batch, train):
        inputs = data_batch[:-1]
        correct_values = torch.cat(data_batch[-1])

        if train:
            molecular_vectors = self.get_molecular_vectors(inputs)
            predicted_values = self.mlp(molecular_vectors)
            return F.mse_loss(predicted_values, correct_values)
        else:
            with torch.no_grad():
                molecular_vectors = self.get_molecular_vectors(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return predicted_values, correct_values

class Trainer(object):
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            labels = [torch.tensor(label, dtype=torch.long).clone().detach() for label in data_batch[-2]]
            labels = tuple(labels)
            data_batch = list(data_batch[:-2]) + [labels]

            if task == 'classification':
                loss = self.model.forward_classifier(data_batch, train=True)
            elif task == 'regression':
                loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()

        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model

    def compute_metrics(self, correct_labels, predicted_scores):
        predicted_labels = (np.array(predicted_scores) >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(correct_labels, predicted_labels).ravel()
        ACC = accuracy_score(correct_labels, predicted_labels)
        PRE = precision_score(correct_labels, predicted_labels)
        SE = recall_score(correct_labels, predicted_labels)
        SP = tn / (tn + fp)
        Gmeans = (SE * SP) ** 0.5
        F1 = f1_score(correct_labels, predicted_labels)
        AUC = roc_auc_score(correct_labels, predicted_scores)
        MCC = matthews_corrcoef(correct_labels, predicted_labels)
        return ACC, PRE, SE, SP, Gmeans, F1, AUC, MCC

    def test_classifier(self, dataset):
        N = len(dataset)
        P, C = [], []
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            labels = [torch.tensor(label, dtype=torch.long).clone().detach() for label in data_batch[-2]]
            labels = tuple(labels)
            data_batch = list(data_batch[:-2]) + [labels]

            predicted_scores, correct_labels = self.model.forward_classifier(data_batch, train=False)
            P.append(predicted_scores)
            C.append(correct_labels)

        P = np.concatenate(P)
        C = np.concatenate(C)
        return self.compute_metrics(C, P)

    def test_regressor(self, dataset):
        N = len(dataset)
        SAE = 0
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            labels = [torch.tensor(label, dtype=torch.float).clone().detach() for label in data_batch[-2]]
            labels = tuple(labels)
            data_batch = list(data_batch[:-2]) + [labels]

            predicted_values, correct_values = self.model.forward_regressor(data_batch, train=False)
            SAE += sum(np.abs(predicted_values - correct_values))

        MAE = SAE / N
        return MAE

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

if __name__ == "__main__":
    task = 'classification'
    dataset = 'b3db'
    radius = 2
    dim = 128
    layer_hidden = 3
    layer_output = 1
    batch_train = 32
    batch_test = 32
    lr = 0.001
    lr_decay = 0.9
    decay_interval = 10
    iteration = 100
    setting = 'experiment_setting'
    nhead = 4
    num_transformer_layers = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        log.info('The code uses a GPU!')
    else:
        log.info('The code uses a CPU...')
    log.info('-'*100)

    log.info(f'Preprocessing the {dataset} dataset.')
    log.info('Just a moment...')
    (dataset_train, dataset_dev, dataset_test, N_fingerprints) = create_datasets(task, dataset, radius, device)
    log.info('-'*100)

    log.info('The preprocess has finished!')
    log.info(f'# of training data samples: {len(dataset_train)}')
    log.info(f'# of development data samples: {len(dataset_dev)}')
    log.info(f'# of test data samples: {len(dataset_test)}')
    log.info('-'*100)

    log.info('Creating a model.')
    torch.manual_seed(1234)
    model = MolecularGraphNeuralNetwork(N_fingerprints, dim, layer_hidden, layer_output, task, nhead, num_transformer_layers).to(device)
    trainer = Trainer(model, lr)
    tester = Tester(model)
    log.info(f'# of model parameters: {sum([np.prod(p.size()) for p in model.parameters()])}')
    log.info('-'*100)

    file_result = f'./output/result--{setting}.txt'
    if task == 'classification':
        result = 'Epoch\tTime(sec)\tLoss_train\tACC_train\tPRE_train\tSE_train\tSP_train\tGmeans_train\tF1_train\tAUC_train\tMCC_train\tACC_test\tPRE_test\tSE_test\tSP_test\tGmeans_test\tF1_test\tAUC_test\tMCC_test'
    elif task == 'regression':
        result = 'Epoch\tTime(sec)\tLoss_train\tMAE_dev\tMAE_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    log.info('Start training.')
    log.info('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):
        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)

        if task == 'classification':
            ACC_train, PRE_train, SE_train, SP_train, Gmeans_train, F1_train, AUC_train, MCC_train = tester.test_classifier(dataset_train)
            ACC_test, PRE_test, SE_test, SP_test, Gmeans_test, F1_test, AUC_test, MCC_test = tester.test_classifier(dataset_test)
            prediction_train = (ACC_train, PRE_train, SE_train, SP_train, Gmeans_train, F1_train, AUC_train, MCC_train)
            prediction_test = (ACC_test, PRE_test, SE_test, SP_test, Gmeans_test, F1_test, AUC_test, MCC_test)
        elif task == 'regression':
            prediction_dev = tester.test_regressor(dataset_dev)
            prediction_test = tester.test_regressor(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            log.info('The training will finish in about %d hours %d minutes.' % (hours, minutes))
            log.info('-'*100)
            log.info(result)

        result = '\t'.join(map(str, [epoch, time, loss_train, *prediction_train, *prediction_test]))
        tester.save_result(result, file_result)

        log.info(f'Epoch: {epoch}, Time: {time:.2f}s, Loss: {loss_train:.4f}')
        log.info(f'Training - ACC: {ACC_train:.4f}, PRE: {PRE_train:.4f}, SE: {SE_train:.4f}, SP: {SP_train:.4f}, Gmeans: {Gmeans_train:.4f}, F1: {F1_train:.4f}, AUC: {AUC_train:.4f}, MCC: {MCC_train:.4f}')
        log.info(f'Testing - ACC: {ACC_test:.4f}, PRE: {PRE_test:.4f}, SE: {SE_test:.4f}, SP: {SP_test:.4f}, Gmeans: {Gmeans_test:.4f}, F1: {F1_test:.4f}, AUC: {AUC_test:.4f}, MCC: {MCC_test:.4f}')