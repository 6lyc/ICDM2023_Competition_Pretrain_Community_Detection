import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor, setup_seed
from dataset import load, load_icdmdata, load_pretrain_data
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.bmm(adj, seq_fts)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret

# Pre-training and Fine-tune
def train(dataset, args, verbose=True):
    
    # pre-training stage
    if args.mode == 'pretrain':
        nb_epochs = 200
        patience = 100
        lr = args.lr
        l2_coef = 0.0
        hid_units = 200
        t = 5
        sparse = False

        adj, diff, features = load_pretrain_data(dataset)

        ft_size = features.shape[1]

        sample_size = args.sample_size
        batch_size = args.batch_size

        lbl_1 = torch.ones(batch_size, sample_size * 2)
        lbl_2 = torch.zeros(batch_size, sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        model = Model(ft_size, hid_units)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

        if torch.cuda.is_available():
            model.cuda()
            lbl = lbl.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        print('pre-train strated!')
        for epoch in range(nb_epochs):

            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
            ba, bd, bf = [], [], []
            for i in idx:
                # ba.append(adj[i: i + sample_size, i: i + sample_size].toarray())
                ba.append(np.eye(sample_size))
                if args.diff_type == 'heat':
                    # bd.append(np.exp(t * (diff[i: i + sample_size, i: i + sample_size].toarray() - 1)))
                    bd.append(np.eye(sample_size))                
                else:
                    bd.append(diff[i: i + sample_size, i: i + sample_size].toarray())
                bf.append(features[i: i + sample_size])

            ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
            bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
            bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

            if sparse:
                ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
                bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
            else:
                ba = torch.FloatTensor(ba)
                bd = torch.FloatTensor(bd)

            bf = torch.FloatTensor(bf)
            idx = np.random.permutation(sample_size)
            shuf_fts = bf[:, idx, :]

            if torch.cuda.is_available():
                bf = bf.cuda()
                ba = ba.cuda()
                bd = bd.cuda()
                shuf_fts = shuf_fts.cuda()

            model.train()
            optimiser.zero_grad()

            logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

            loss = b_xent(logits, lbl)

            loss.backward()
            optimiser.step()
            if verbose:
                    print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))
            
            if loss < best:    
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), './model_pkl/pre-train/model_exp{}.pkl'.format(args.exp))
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                if verbose:
                    print('Early stopping!')
                break
    
    # fine-tune stage
    elif args.mode == 'ft':
        ft_epochs = 20
        patience = 15
        lr = args.lr
        l2_coef = 0.0
        hid_units = 200
        t = 5
        sparse = False

        adj, diff, features = load_icdmdata(dataset)

        ft_size = features.shape[1]

        sample_size = args.sample_size
        batch_size = args.batch_size

        lbl_1 = torch.ones(batch_size, sample_size * 2)
        lbl_2 = torch.zeros(batch_size, sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        model = Model(ft_size, hid_units)
        # load the pre-training model
        model.load_state_dict(torch.load('./model_pkl/pre-train/model_exp{}.pkl'.format(args.exp), map_location='cpu'))

        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

        if torch.cuda.is_available():
            model.cuda()
            lbl = lbl.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        print('fine-tune strated!')
        for epoch in range(ft_epochs):

            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
            ba, bd, bf = [], [], []
            for i in idx:
                # ba.append(adj[i: i + sample_size, i: i + sample_size].toarray())
                ba.append(np.eye(sample_size))
                if args.diff_type == 'heat':
                    # bd.append( np.exp(t * (diff[i: i + sample_size, i: i + sample_size].toarray() - 1)))
                    bd.append(np.eye(sample_size))                
                else:
                    bd.append(diff[i: i + sample_size, i: i + sample_size].toarray())
                bf.append(features[i: i + sample_size])

            ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
            bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
            bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

            if sparse:
                ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
                bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
            else:
                ba = torch.FloatTensor(ba)
                bd = torch.FloatTensor(bd)

            bf = torch.FloatTensor(bf)
            idx = np.random.permutation(sample_size)
            shuf_fts = bf[:, idx, :]

            if torch.cuda.is_available():
                bf = bf.cuda()
                ba = ba.cuda()
                bd = bd.cuda()
                shuf_fts = shuf_fts.cuda()

            model.train()
            optimiser.zero_grad()

            logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

            loss = b_xent(logits, lbl)

            loss.backward()
            optimiser.step()

            if loss < best:
                if verbose:
                    print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), './model_pkl/fine-tune/model_exp{}.pkl'.format(args.exp))
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                if verbose:
                    print('Early stopping!')
                break

        if verbose:
            print('Loading {}th epoch: '.format(best_t), best)
        model.load_state_dict(torch.load('./model_pkl/fine-tune/model_exp{}.pkl'.format(args.exp)))
        print('fine-tune model loaded!')

        sparse = True
        if sparse:
            adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
            diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

        features = torch.FloatTensor(features[np.newaxis])
        adj = torch.unsqueeze(adj, 0)
        diff = torch.unsqueeze(diff, 0)

        if torch.cuda.is_available():
            features = features.cuda()
            adj = adj.cuda()
            diff = diff.cuda()

        print('test started!')
        embeds, _ = model.embed(features, adj, diff, sparse, None)
        print('embedding finished!')

        # community detection
        embedding = embeds.squeeze()
        
        X_np = embedding.cpu().numpy()

        # create KMeans model, the number of clusters is 350
        print('clustering started!')
        kmeans = KMeans(n_clusters=350, random_state=0).fit(X_np)
        print('clustering finished!')
        # get node clustering labels
        labels = kmeans.labels_
        labels = labels + 1  

        # save the labels to txt
        np.savetxt('./result/GTML-中山大学-exp{}.txt'.format(args.exp), labels, fmt='%d')
    
    # local ft and test
    else:
        ft_epochs = 100
        patience = 10
        lr = args.lr
        l2_coef = 0.0
        hid_units = 200
        t = 5
        sparse = False

        adj, diff, features, labels, _, _, _ = load(dataset)

        # reduce dim
        pca = PCA(n_components=100)
        features = pca.fit_transform(features)
        ft_size = features.shape[1]

        sample_size = args.sample_size
        batch_size = args.batch_size

        lbl_1 = torch.ones(batch_size, sample_size * 2)
        lbl_2 = torch.zeros(batch_size, sample_size * 2)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        model = Model(ft_size, hid_units)
        # load the pre-training model
        model.load_state_dict(torch.load('./model_pkl/pre-train/model_exp{}.pkl'.format(args.exp), map_location='cpu'))

        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

        if torch.cuda.is_available():
            model.cuda()
            lbl = lbl.cuda()

        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        cnt_wait = 0
        best = 1e9
        best_t = 0

        sparse = False
        if sparse:
            adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
            diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

        features = torch.FloatTensor(features[np.newaxis])
        adj = torch.FloatTensor(adj[np.newaxis])
        diff = torch.FloatTensor(diff[np.newaxis])

        if torch.cuda.is_available():
            features = features.cuda()
            adj = adj.cuda()
            diff = diff.cuda()

        print('local fine-tune strated!')
        for epoch in range(ft_epochs):

            embeds, _ = model.embed(features, adj, diff, sparse, None)

            # community detection
            embedding = embeds.squeeze()
            
            X_np = embedding.cpu().numpy()

            # create KMeans model, the number of clusters is 7
            kmeans = KMeans(n_clusters=args.local_nclusters, random_state=0).fit(X_np)

            pred_labels = kmeans.labels_
            
            ari = adjusted_rand_score(labels, pred_labels)
            print('epoch, ari: ', epoch, ari)
            
            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
            ba, bd, bf = [], [], []
            for i in idx:
                ba.append(torch.squeeze(adj)[i: i + sample_size, i: i + sample_size])
                bd.append(torch.squeeze(diff)[i: i + sample_size, i: i + sample_size])
                bf.append(torch.squeeze(features)[i: i + sample_size])

            ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
            bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
            bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

            if sparse:
                ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
                bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
            else:
                ba = torch.FloatTensor(ba)
                bd = torch.FloatTensor(bd)

            bf = torch.FloatTensor(bf)
            idx = np.random.permutation(sample_size)
            shuf_fts = bf[:, idx, :]

            if torch.cuda.is_available():
                bf = bf.cuda()
                ba = ba.cuda()
                bd = bd.cuda()
                shuf_fts = shuf_fts.cuda()

            model.train()
            optimiser.zero_grad()

            logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

            loss = b_xent(logits, lbl)

            loss.backward()
            optimiser.step()
            if verbose:
                print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), './model_pkl/local-fine-tune/model_exp{}.pkl'.format(args.exp))
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                if verbose:
                    print('Early stopping!')
                break

        if verbose:
            print('Loading {}th epoch: '.format(best_t), best)
        model.load_state_dict(torch.load('./model_pkl/local-fine-tune/model_exp{}.pkl'.format(args.exp)))
        print('local-fine-tune model loaded!')


        print('test started!')
        embeds, _ = model.embed(features, adj, diff, sparse, None)
        print('embedding finished!')

        # community detection
        embedding = embeds.squeeze()
        
        X_np = embedding.cpu().numpy()

        # create KMeans model, the number of clusters is 7
        print('clustering started!')
        kmeans = KMeans(n_clusters=args.local_nclusters, random_state=0).fit(X_np)
        print('clustering finished!')

        pred_labels = kmeans.labels_
        
        ari = adjusted_rand_score(labels, pred_labels)
        print('best model ari: ', ari)



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    import argparse
    parser = argparse.ArgumentParser(
        description='train_community_discovery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, default='pretrain', help="pretrain, ft or local ft")
    parser.add_argument('--diff_type', type=str, default='heat', help="heat or diff")
    parser.add_argument('--local_nclusters', type=int, default=7)
    parser.add_argument('--local_test_dataset', type=str, default="citeseer")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--exp', type=int, default=1)

    args = parser.parse_args()

    setup_seed(2024)

    if args.device == -1:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] =""
    else:
        torch.cuda.set_device(args.device)

    # 'cora', 'citeseer', 'arxiv_all_nodes_heat', 'icdmdata_all_nodes_heat', 'icdmdata_all_nodes_ppr'
    if args.mode == 'pretrain': 
        dataset = 'arxiv_all_nodes_heat'
    elif args.mode == 'ft':
        if args.diff_type == 'heat':
            dataset = 'icdmdata_all_nodes_heat'
        else:
            dataset = 'icdmdata_all_nodes_ppr'
    else:
        dataset = args.local_test_dataset
    
    for __ in range(1):
        train(dataset, args)
