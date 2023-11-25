import os
import numpy as np
import dgl
from sklearn.decomposition import PCA
from ogb.nodeproppred import DglNodePropPredDataset

# Function to load 'ogbn-arxiv' dataset
def load_ogbn_arxiv():
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='./dataset')  
    graph, _ = dataset[0]
    
    # Reduce node features' dimensionality to 100 using PCA
    pca = PCA(n_components=100)
    node_feat_reduced = pca.fit_transform(graph.ndata['feat'])

    return graph, node_feat_reduced

# Function to save edge and node features to text files
def save_to_txt(edge_file, node_file, graph, node_features):
    # get the edge index 
    src, dst = graph.edges()

    # save edge txt
    np.savetxt(edge_file, np.hstack((src[:,None], dst[:,None])), 
            fmt='%d', delimiter=',', header='', comments='')
            
    # save feature txt
    np.savetxt(node_file, node_features, 
            fmt='%.10f', delimiter=',', header='', comments='')

if __name__ == '__main__':
    # Load 'ogbn-arxiv' dataset
    graph, node_feat_reduced = load_ogbn_arxiv()

    # Specify file paths for saving edge and node features
    edge_file_path = './icdm2023_session1_test/ogbn_arxiv_edge.txt'
    node_file_path = './icdm2023_session1_test/ogbn_arxiv_node_feat.txt'

    # Save edge and node features to txt files
    save_to_txt(edge_file_path, node_file_path, graph, node_feat_reduced)
