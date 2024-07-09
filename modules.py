import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class FeatureGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, alpha=0.2, concat=True):
        super(FeatureGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x):
        h = torch.matmul(x, self.W)
        batch_size, N, _ = h.size()
        a_input = self._make_attention_input(h)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        attention = torch.softmax(e, dim=2)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return torch.relu(h_prime)
        else:
            return h_prime

    def _make_attention_input(self, x):
        B, N, D = x.size()
        x = x.repeat(1, 1, N).view(B, N * N, D)
        x_ = x.view(B, N, N, D)
        x_ = x_.repeat(1, 1, 1, N).view(B, N, N, N, D)
        x_ = x_.permute(0, 3, 1, 2, 4)
        combined = torch.cat([x_.view(B, N * N * N, D), x], dim=-1)
        return combined.view(B, N, N, 2 * D)

    # Ensure the reshaping dimensions match the input tensor's size
    # Calculate the correct dimensions based on the input tensor

    def _make_attention_input(self, x):
        B, N, D = x.size()
        print(f"Original x shape: {x.shape}")

        x_repeat = x.repeat(1, 1, N)
        x_repeat = x_repeat.view(B, N, N, D)
        print(f"x_repeat shape: {x_repeat.shape}")

        x_ = x_repeat.repeat(1, N, 1, 1)
        x_ = x_.view(B, N, N, N, D)
        print(f"x_ shape: {x_.shape}")

        x_ = x_.permute(0, 1, 3, 2, 4)
        print(f"x_ permuted shape: {x_.shape}")

        combined = torch.cat([x_.view(B, N * N, D), x.view(B, N, D).repeat(1, N, 1)], dim=-1)
        print(f"Combined shape before view: {combined.shape}")

        combined = combined.view(B, N, N, 2 * D)
        print(f"Combined shape after view: {combined.shape}")

        return combined
