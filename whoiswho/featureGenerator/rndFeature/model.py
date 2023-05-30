import os
import sys
import torch
import torch.nn as nn
sys.path.append('../../../')
from whoiswho.config import configs

from typing import Union
from torch import Tensor
import torch.nn.functional as F

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
        # print(sim_vec)
        sim_vec = sim_vec.unsqueeze(-1)
        # print(sim_vec.size())
        pooling_value = torch.exp((- ((sim_vec - self.mu) ** 2) / (self.sigma ** 2) / 2))
        # print(pooling_value.size())
        pooling_sum = torch.sum(pooling_value, 1)
        # print(pooling_sum.size())
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 0)
        # print(log_pooling_sum.size())
        # exit()
        # log_pooling_sum = self.miAttention(log_pooling_sum)
        # print(log_pooling_sum.size())
        return log_pooling_sum

    def get_intersect_matrix(self, paper_embedding, per_embedding):
        sim_vec = self.each_field_model(paper_embedding, per_embedding)
        return sim_vec
        # return log_pooling_sum

    def forward(self, paper_embedding, per_embedding):
        paper_embedding = torch.nn.functional.normalize(paper_embedding, p=2, dim=1)
        per_embedding = torch.nn.functional.normalize(per_embedding, p=2, dim=1)  # 将每个维度都除以其范数
        # print(paper_cls_embedding.size(), author_cls_embedding.size(), author_per_cls_embedding.size())

        whole_sim = self.get_intersect_matrix(paper_embedding, per_embedding)
        # print(whole_sim.size())
        return whole_sim

class learning2RankRawNew(nn.Module):
    def __init__(self, mlp_dim):
        """
        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(learning2RankRawNew, self).__init__()
        self.cos_vec = cosMlp()
        self.euc_vec = eucMlp()
        self.str_vec = strMlp()

        self.learning2Rank = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid()
        )

    def forward(self,str_feature, whole_sim=None, cos_feature=None, euc_feature=None):
        str_feature = str_feature.float()
        str2vec = self.str_vec(str_feature.unsqueeze(0))
        total_vec = str2vec
        if whole_sim is not None:
            whole_sim = whole_sim.float()
            whole_sim = whole_sim.unsqueeze(0)
            total_vec = torch.cat((total_vec, whole_sim), 1)
        if cos_feature is not None:
            cos_feature = cos_feature.float()
            cos2vec = self.cos_vec(cos_feature.unsqueeze(0))
            total_vec = torch.cat((total_vec, cos2vec), 1)
        if euc_feature is not None:
            euc_feature = euc_feature.float()
            euc2vec = self.euc_vec(euc_feature.unsqueeze(0))
            total_vec = torch.cat((total_vec, euc2vec), 1)

        output = self.learning2Rank(total_vec)
        return output

    def predict_proba(self, str_feature, whole_sim=None, cos_feature=None, euc_feature=None):
        return self.forward(str_feature, whole_sim, cos_feature, euc_feature)
'''
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
'''