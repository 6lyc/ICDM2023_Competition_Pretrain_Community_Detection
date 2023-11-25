import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
from scipy.sparse.linalg import inv as s_inv
from scipy.sparse.linalg import expm
import scipy.sparse as sp
import math
import random


def setup_seed(seed):
    # random
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    torch.cuda.manual_seed(seed)
    # 多GPU（涵盖了torch.cuda.manual_seed()）
    torch.cuda.manual_seed_all(seed)
    # numpy
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

def compute_ppr_sparse(graph: nx.Graph, alpha=0.2, self_loop=True):

    # turn to the sparse graph
    A = nx.to_scipy_sparse_array(graph, format='csc')  
    
    if self_loop:
        A = A + sp.eye(A.shape[0])
        
    D_inv = sp.diags(1 / A.sum(axis=1), format='csr')  
    
    At = D_inv @ A

    B = sp.eye(A.shape[0], format='csc') - (1 - alpha) * At
    I = sp.identity(A.shape[0], format='csc') 
    B_inv = sp.identity(A.shape[0], format='csr')
    print('iterated start!')
    for i in range(1):
        print('epoch-', i)
        B_inv = B_inv @ (2*I - B @ B_inv)

    PPR_diff = alpha * B_inv

    return PPR_diff


def compute_heat(graph: nx.Graph, t=5, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))

def compute_heat_sparse(graph: nx.Graph, t=5, self_loop=True):
    A = nx.to_scipy_sparse_array(graph, format='csr')
    if self_loop:
        A = A + sp.eye(A.shape[0])
    
    D_inv = sp.diags(1 / A.sum(axis=1), format='csc')
    
    C = A @ D_inv

    return C


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)