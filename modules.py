import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ConvLayer, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class FeatureAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, embed_dim, use_gatv2):
        super(FeatureAttentionLayer, self).__init__()
        self.gat = GATConv(in_dim, out_dim, heads=1, concat=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x):
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.gat(x, edge_index=None)
        x = self.elu(x)
        x = x.view(B, N, -1)
        return x

class TemporalAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha, embed_dim, use_gatv2):
        super(TemporalAttentionLayer, self).__init__()
        self.gat = GATConv(in_dim, out_dim, heads=1, concat=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x):
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.gat(x, edge_index=None)
        x = self.elu(x)
        x = x.view(B, N, -1)
        return x

class GRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        return self.gru(x)

class Forecasting_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ReconstructionModel(nn.Module):
    def __init__(self, window_size, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(ReconstructionModel, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, window_size * output_dim))
        self.model = nn.Sequential(*layers)
        self.window_size = window_size
        self.output_dim = output_dim

    def forward(self, x):
        return self.model(x).view(-1, self.window_size, self.output_dim)
