import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(csv_path="./data/Water_Discharge_Data.csv"):
    df = pd.read_csv(csv_path)

    # Drop timestamp and convert to numpy
    discharge_data = df.iloc[:, 1:].values  # shape: [T, N]
    discharge_data = discharge_data.T       # new shape: [N, T] --> N nodes (stations), each with T features

    # Normalize discharge across time (axis=1)
    scaler = MinMaxScaler()
    discharge_data = scaler.fit_transform(discharge_data)

    features = torch.FloatTensor(discharge_data)  # shape [N, T]
    
    features = discharge_data[:, :-1]  # [N, T-1] — each station is a node
    labels = discharge_data[:, 1:]     # [N, T-1] — next-day for each node (or pick just one if needed)


    # Load adjacency matrix (shape [N, N])
    adj = sp.load_npz('./data/adj.npz')
    adj = sparse_mx_to_torch_sparse_tensor(normalize(adj + sp.eye(adj.shape[0])))

    # Split
    num_samples = features.shape[0]
    idx_train = torch.LongTensor(range(int(num_samples * 0.6)))
    idx_val = torch.LongTensor(range(int(num_samples * 0.6), int(num_samples * 0.8)))
    idx_test = torch.LongTensor(range(int(num_samples * 0.8), num_samples))

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
