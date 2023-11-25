import torch
from dgl.data import CitationGraphDataset
import dgl
from utils import preprocess_features, normalize_adj
from sklearn.preprocessing import MinMaxScaler
from utils import compute_ppr, compute_ppr_sparse, compute_heat_sparse
import scipy.sparse as sp
import networkx as nx
import numpy as np
import pandas as pd
import os
import pickle


def download(dataset):
    if dataset == 'cora':
        return CitationGraphDataset(name=dataset)
    elif dataset == 'citeseer' or 'pubmed':
        return CitationGraphDataset(name=dataset)
    else:
        return None


def load(dataset):
    datadir = os.path.join('data', dataset)

    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(dataset)
        adj = nx.to_numpy_array(ds.graph)
        diff = compute_ppr(ds.graph, 0.2)
        feat = ds.features[:]
        labels = ds.labels[:]

        idx_train = np.argwhere(ds.train_mask == 1).reshape(-1)
        idx_val = np.argwhere(ds.val_mask == 1).reshape(-1)
        idx_test = np.argwhere(ds.test_mask == 1).reshape(-1)
        
        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/diff.npy', diff)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
        np.save(f'{datadir}/idx_train.npy', idx_train)
        np.save(f'{datadir}/idx_val.npy', idx_val)
        np.save(f'{datadir}/idx_test.npy', idx_test)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        diff = np.load(f'{datadir}/diff.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')
        idx_train = np.load(f'{datadir}/idx_train.npy')
        idx_val = np.load(f'{datadir}/idx_val.npy')
        idx_test = np.load(f'{datadir}/idx_test.npy')

    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                      for e in epsilons])]

        diff[diff < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff)
        diff = scaler.transform(diff)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, diff, feat, labels, idx_train, idx_val, idx_test


def load_icdmdata(dataset):
    print('load dataset: ', dataset)
    datadir = os.path.join('./icdm2023_session1_test/data', dataset)
    
    if not os.path.exists(datadir):
        # read the node feature data
        features = pd.read_csv('./icdm2023_session1_test/icdm2023_session1_test_node_feat.txt', header=None)
        features = np.array(features)
        print('icdm features read!')

        # create a directed graph
        if not os.path.exists('./icdm2023_session1_test/data/graph_all.pkl'):
            # read the edge data
            edges = pd.read_csv('./icdm2023_session1_test/icdm2023_session1_test_edge.txt', header=None, names=["src", "dst"], sep=",")
            src = edges["src"].tolist()
            dst = edges["dst"].tolist()
            print('icdm edges read!')

            node_nums = max(max(src), max(dst))
            nodes = list(range(node_nums + 1))
            g = nx.Graph()
            g.add_nodes_from(nodes)
            g.add_edges_from(zip(src, dst))
            print('graph created!')

            # save the nx graph data
            with open("./icdm2023_session1_test/data/graph_all.pkl", "wb") as file:
                pickle.dump(g, file)

        else:
            # load the nx graph data
            with open("./icdm2023_session1_test/data/graph_all.pkl", "rb") as file:
                g = pickle.load(file)
                print('load icdm graph!')

        g_feat = torch.from_numpy(features) 

        os.makedirs(datadir)
        adj = nx.to_scipy_sparse_array(g)
        # diff = compute_ppr_sparse(g, 0.2)
        diff = compute_heat_sparse(g)
        feat = g_feat[:]
        
        sp.save_npz(f'{datadir}/adj.npz', adj)
        sp.save_npz(f'{datadir}/diff.npz', diff)
        np.save(f'{datadir}/feat.npy', feat)
        
    else:
        adj = sp.load_npz(f'{datadir}/adj.npz')
        print('load icdm adj!')
        diff = sp.load_npz(f'{datadir}/diff.npz')
        print('load icdm diff!')
        feat = np.load(f'{datadir}/feat.npy')
        print('load icdm feature!')

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    return adj, diff, feat

def load_pretrain_data(dataset):
    print('load dataset: ', dataset)
    datadir = os.path.join('./icdm2023_session1_test/data', dataset)
    
    if not os.path.exists(datadir):
        
        features = pd.read_csv('./icdm2023_session1_test/ogbn_arxiv_node_feat.txt', header=None)
        features = np.array(features)
        print('arxiv features read!')

        if not os.path.exists('./icdm2023_session1_test/data/arxiv_graph_all.pkl'):
            edges = pd.read_csv('./icdm2023_session1_test/ogbn_arxiv_edge.txt', header=None, names=["src", "dst"], sep=",")
            src = edges["src"].tolist()
            dst = edges["dst"].tolist()
            print('arxiv edges read!')

            node_nums = max(max(src), max(dst))
            nodes = list(range(node_nums + 1))
            g = nx.Graph()
            g.add_nodes_from(nodes)
            g.add_edges_from(zip(src, dst))
            print('arxiv graph created!')

            with open("./icdm2023_session1_test/data/ogbn_arxiv_graph_all.pkl", "wb") as file:
                pickle.dump(g, file)

        else:
            with open("./icdm2023_session1_test/data/ogbn_arxiv_graph_all.pkl", "rb") as file:
                g = pickle.load(file)
                print('load arxiv graph!')

        g_feat = torch.from_numpy(features) 

        os.makedirs(datadir)
        adj = nx.to_scipy_sparse_array(g)
        # diff = compute_ppr_sparse(g, 0.2)
        diff = compute_heat_sparse(g)
        feat = g_feat[:]
        
        sp.save_npz(f'{datadir}/adj.npz', adj)
        sp.save_npz(f'{datadir}/diff.npz', diff)
        np.save(f'{datadir}/feat.npy', feat)
        
    else:
        adj = sp.load_npz(f'{datadir}/adj.npz')
        print('load arxiv adj!')
        diff = sp.load_npz(f'{datadir}/diff.npz')
        print('load arxiv diff!')
        feat = np.load(f'{datadir}/feat.npy')
        print('load arxiv feature!')

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    return adj, diff, feat

if __name__ == '__main__':
    load_pretrain_data('arxiv_all_nodes_heat')
