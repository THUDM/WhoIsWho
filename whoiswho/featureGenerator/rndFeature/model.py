import os
import sys
import torch
import torch.nn as nn
sys.path.append('../../../')
from whoiswho.config import configs
from math import cos, pi
from typing import Union
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, GATConv

gat_parameters = {'lr':1e-3
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
              ,'layer_heads':[4,1]
             }
sage_parameters = {'lr':1e-3
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }

class strMlp(nn.Module):
    def __init__(self):
        super(strMlp, self).__init__()
        self.n_bins = 36
        self.strDenseLayer = nn.Sequential(
            nn.Linear(self.n_bins, self.n_bins),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins)
        )

    def forward(self, raw_feature):
        # print(raw_feature.size())
        str2vec = self.strDenseLayer(raw_feature)
        return str2vec

class bertSimiMlp(nn.Module):
    def __init__(self):
        super(bertSimiMlp, self).__init__()
        self.n_bins = 41
        self.bertSimiDenseLayer = nn.Sequential(
            nn.Linear(self.n_bins, self.n_bins),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins)
        )

    def forward(self, raw_feature):
        bertSimi2vec = self.bertSimiDenseLayer(raw_feature)
        return bertSimi2vec

class cosMlp(nn.Module):
    def __init__(self):
        super(cosMlp, self).__init__()
        self.n_bins = 41
        self.cosDenseLayer = nn.Sequential(
            nn.Linear(self.n_bins, self.n_bins),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins)
        )

    def forward(self, raw_feature):
        cos2vec = self.cosDenseLayer(raw_feature)
        return cos2vec

class eucMlp(nn.Module):
    def __init__(self):
        super(eucMlp, self).__init__()
        self.n_bins = 41
        self.eucDenseLayer = nn.Sequential(
            nn.Linear(self.n_bins, self.n_bins),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.n_bins, self.n_bins)
        )

    def forward(self, raw_feature):
        euc2vec = self.eucDenseLayer(raw_feature)
        return euc2vec

def kernal_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    print(l_mu)
    return l_mu

def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    print(l_sigma)
    return l_sigma

def infonce(logits,tau,label=None): #(1,authors)
    # apply temperature
    # logits /= tau
    # label: positive key indicators
    if label:
        label = torch.tensor(label, dtype=torch.long).unsqueeze(0).cuda()
    else:
        label = torch.zeros(1, dtype=torch.long).cuda()

    loss = nn.CrossEntropyLoss().cuda()(logits, label)
    return loss

def HR(truth, pred, N):
    """
    Hit Ratio
    truth: the index of real target (from 0 to len(truth)-1)
    pred: rank list
    N: top-N elements of rank list
    """
    top_N = pred.squeeze()[:N]
    hit_N = 1.0 if truth in top_N else 0.0
    return hit_N

def MRR(truth, pred, N):
    """
    Mean Reciprocal Rank
    truth: the index of real target (from 0 to len(truth)-1)
    pred: rank list
    N: top-N elements of rank list
    """
    top_N = pred.squeeze()[:N]
    if truth not in top_N:
        rr_N = 0.0
    else:
        i = top_N.cpu().numpy().tolist().index(truth)
        rr_N = 1 / (i + 1) # index从0开始
    return rr_N

def adjust_learning_rate(optimizer, current_epoch,max_epoch,lr_min=0,lr_max=0.1,warmup=True):
    warmup_epoch = 5 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class bertEmbeddingLayer(nn.Module):
    def __init__(self, bertModel):
        super(bertEmbeddingLayer, self).__init__()
        self.bertModel = bertModel

    def forward(self, ins_input_ids, ins_token_type_ids, ins_attention_mask, ins_position_ids, ins_position_ids_second):
        _, pooled_output = self.bertModel.bert.forward(
            input_ids=ins_input_ids,
            token_type_ids=ins_token_type_ids,
            attention_mask=ins_attention_mask,
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=ins_position_ids,
            position_ids_second=ins_position_ids_second
        )
        return _, pooled_output


class matchingModel(nn.Module):
    def __init__(self, device):
        """
        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(matchingModel, self).__init__()
        self.n_bins = configs["raw_feature_len"]
        self.device = device
        # print(self.device)
        self.mu = torch.FloatTensor(kernal_mus(self.n_bins)).to(device, non_blocking=True)
        self.sigma = torch.FloatTensor(kernel_sigmas(self.n_bins)).to(device, non_blocking=True)

        self.mu = self.mu.view(1, 1, self.n_bins)
        self.sigma = self.sigma.view(1, 1, self.n_bins)

    def each_field_model(self, paper_embedding, per_embedding):
        sim_vec = per_embedding @ paper_embedding.transpose(1, 0)
        sim_vec = sim_vec.unsqueeze(-1)
        pooling_value = torch.exp((- ((sim_vec - self.mu) ** 2) / (self.sigma ** 2) / 2))
        pooling_sum = torch.sum(pooling_value, 1)

        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 0)

        return log_pooling_sum

    def get_intersect_matrix(self, paper_embedding, per_embedding):
        sim_vec = self.each_field_model(paper_embedding, per_embedding)
        return sim_vec
        # return log_pooling_sum

    def forward(self, paper_embedding, per_embedding):
        paper_embedding = torch.nn.functional.normalize(paper_embedding, p=2, dim=1)
        per_embedding = torch.nn.functional.normalize(per_embedding, p=2, dim=1)
        # print(paper_cls_embedding.size(), author_cls_embedding.size(), author_per_cls_embedding.size())

        whole_sim = self.get_intersect_matrix(paper_embedding, per_embedding)
        # print(whole_sim.size())
        return whole_sim


class SAGE(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , layer_heads = []
                 , batchnorm=True):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=layer_heads[0], concat=True))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[0]))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels*layer_heads[i-1], hidden_channels, heads=layer_heads[i], concat=True))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels*layer_heads[i-1]))
        self.convs.append(GATConv(hidden_channels*layer_heads[num_layers-2]
                          , out_channels
                          , heads=layer_heads[num_layers-1]
                          , concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class matchingModel_gnn(nn.Module):
    def __init__(self, device):
        """
        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(matchingModel, self).__init__()
        self.n_bins = 41
        self.device = device

        self.mu = torch.FloatTensor(kernal_mus(41)).to(device, non_blocking=True)
        self.sigma = torch.FloatTensor(kernel_sigmas(41)).to(device, non_blocking=True)

        self.mu = self.mu.view(1, 1, self.n_bins)
        self.sigma = self.sigma.view(1, 1, self.n_bins)
        # temperature
        self.T = 0.1
        ''' 
        Train gnn to make the obtained strcture feature more effective, 
        set less parameters for the linear layers. 
        '''
        self.scoring_layer = nn.Linear(self.n_bins, 1)


    def each_field_model(self, paper_embedding, per_embedding):
        sim_vec = per_embedding @ paper_embedding.transpose(1, 0)
        sim_vec = sim_vec.unsqueeze(-1)
        pooling_value = torch.exp((- ((sim_vec - self.mu) ** 2) / (self.sigma ** 2) / 2))
        pooling_sum = torch.sum(pooling_value, 1)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 0,keepdim=True)

        return log_pooling_sum

    def get_intersect_matrix(self, paper_embedding, per_embedding):
        sim_vec = self.each_field_model(paper_embedding, per_embedding)
        return sim_vec


    def forward(self, paper_embedding, authors_embedding,test=False):
        #authors_embedding (authors,nodes,128)
        paper_embedding = torch.nn.functional.normalize(paper_embedding, p=2, dim=1)
        whole_sim_feat=[]
        if test: # author_graph_emb [(graph_embs,name)...]
            for author_graph_emb in authors_embedding: #(author_embs,author_name)
                author_graph_emb = torch.nn.functional.normalize(author_graph_emb[0], p=2, dim=1)  # 将每个维度都除以其范数
                paper_author_sim_feat = self.get_intersect_matrix(paper_embedding, author_graph_emb)
                whole_sim_feat.append(paper_author_sim_feat)
            whole_sim_feat = torch.cat(whole_sim_feat) #(authors,41)
            logits=self.scoring_layer(whole_sim_feat)  #(authors,1)
        else:
            for author_graph_emb in authors_embedding: #(author_embs,author_name)
                author_graph_emb = torch.nn.functional.normalize(author_graph_emb, p=2, dim=1)  # 将每个维度都除以其范数
                paper_author_sim_feat = self.get_intersect_matrix(paper_embedding, author_graph_emb)
                whole_sim_feat.append(paper_author_sim_feat)
            whole_sim_feat = torch.cat(whole_sim_feat) #(authors,41)
            logits=self.scoring_layer(whole_sim_feat)  #(authors,1)

        return logits.transpose(1, 0)  #(1,authors)