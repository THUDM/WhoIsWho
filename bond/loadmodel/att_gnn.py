import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ATTGNN(nn.Module):

    def __init__(self, layer_shape):
        super(ATTGNN, self).__init__()
        self.layer_shape = layer_shape
        self.conv1 = GATConv(layer_shape[0], layer_shape[1], heads=1)
        self.conv2 = GATConv(layer_shape[1], layer_shape[2], heads=1)
        self.clas_layer = nn.Parameter(torch.FloatTensor(layer_shape[-2], layer_shape[-1]))
        self.bias = nn.Parameter(torch.FloatTensor(1, layer_shape[-1]))

        nn.init.xavier_uniform_(self.clas_layer.data, gain=1.414)
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)
    
    def forward(self, ft_list, adj_tensor, edge_attr=None):
        x_list = self.conv1(ft_list, adj_tensor, edge_attr)
        x_list = self.non_linear(x_list)
        x_list = self.dropout_ft(x_list, 0.5)

        x_list = self.conv2(x_list, adj_tensor, edge_attr)
        logits = torch.mm(x_list, self.clas_layer)+self.bias

        embd = x_list
        return logits, embd
        
    def non_linear(self, x_list):
        y_list = F.elu(x_list)
        return y_list
    
    def dropout_ft(self, x_list, dropout):
        y_list = F.dropout(x_list, dropout, training=self.training)
        return y_list
