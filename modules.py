import torch
import torch.nn as nn

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
    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def _make_attention_input(self, v):
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        # Debugging statements
        print(f"FeatureAttentionLayer _make_attention_input - v.shape: {v.shape}")
        print(f"FeatureAttentionLayer _make_attention_input - combined.shape: {combined.shape}")

        combined_size = combined.size(2)
        return combined.view(v.size(0), K, K, combined_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        if self.use_gatv2:
            a_input = self._make_attention_input(x)
            print(f"FeatureAttentionLayer forward - a_input.shape before lin: {a_input.shape}")
            a_input = a_input.view(-1, a_input.size(3))  # Flatten for linear layer
            print(f"FeatureAttentionLayer forward - a_input.shape after flatten: {a_input.shape}")
            a_input = self.leakyrelu(self.lin(a_input))
            a_input = a_input.view(x.size(0), self.n_features, self.n_features, -1)  # Reshape back
            print(f"FeatureAttentionLayer forward - a_input.shape after lin: {a_input.shape}")
            e = torch.matmul(a_input, self.a).squeeze(3)
        else:
            Wx = self.lin(x)
            a_input = self._make_attention_input(Wx)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)

        if self.use_bias:
            e += self.bias

        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)


class TemporalAttentionLayer(nn.Module):
    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def _make_attention_input(self, v):
        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)
        blocks_alternating = v.repeat(1, K, 1)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        # Debugging statements
        print(f"TemporalAttentionLayer _make_attention_input - v.shape: {v.shape}")
        print(f"TemporalAttentionLayer _make_attention_input - combined.shape: {combined.shape}")

        combined_size = combined.size(2)
        return combined.view(v.size(0), K, K, combined_size)

    def forward(self, x):
        if self.use_gatv2:
            a_input = self._make_attention_input(x)
            print(f"TemporalAttentionLayer forward - a_input.shape before lin: {a_input.shape}")
            a_input = a_input.view(-1, a_input.size(3))  # Flatten for linear layer
            print(f"TemporalAttentionLayer forward - a_input.shape after flatten: {a_input.shape}")
            a_input = self.leakyrelu(self.lin(a_input))
            a_input = a_input.view(x.size(0), self.window_size, self.window_size, -1)  # Reshape back
            print(f"TemporalAttentionLayer forward - a_input.shape after lin: {a_input.shape}")
            e = torch.matmul(a_input, self.a).squeeze(3)
        else:
            Wx = self.lin(x)
            a_input = self._make_attention_input(Wx)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)

        if self.use_bias:
            e += self.bias

        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h = self.sigmoid(torch.matmul(attention, x))

        return h


class GRULayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        out, h = self.gru(x)
        out, h = out[-1, :, :], h[-1, :, :]
        return out, h

class RNNDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out

class ReconstructionModel(nn.Module):
    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out

class Forecasting_Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)
