import torch
import torch.nn as nn
import numpy as np
from modules import (
    ConvLayer,
    FeatureAttentionLayer,
    TemporalAttentionLayer,
    GRULayer,
    Forecasting_Model,
    ReconstructionModel,
)

class MTAD_GAT(nn.Module):
    def __init__(
            self,
            n_features,
            window_size,
            out_dim,
            kernel_size=7,
            feat_gat_embed_dim=None,
            time_gat_embed_dim=None,
            use_gatv2=True,
            gru_n_layers=1,
            gru_hid_dim=150,
            forecast_n_layers=1,
            forecast_hid_dim=150,
            recon_n_layers=1,
            recon_hid_dim=150,
            dropout=0.2,
            alpha=0.2
    ):
        super(MTAD_GAT, self).__init__()
        self.window_size = window_size
        self.conv = ConvLayer(n_features=n_features, kernel_size=kernel_size, out_channels=64)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim,
                                                   use_gatv2)
        self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        if x.dim() == 2:  # If the input tensor has only 2 dimensions
            x = x.unsqueeze(1)  # Add an extra dimension to make it 3D
        x = x.permute(0, 2, 1)  # Permute to match Conv1d input format
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)
        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)  # Hidden state for last timestamp
        predictions = self.forecasting_model(h_end)
        recons = self.recon_model(h_end)
        return predictions, recons

    def sliding_window_inference(self, data, step_size=1):
        all_predictions = []
        all_recons = []
        for start in range(0, data.shape[0] - self.window_size + 1, step_size):
            end = start + self.window_size
            window_data = data[start:end].unsqueeze(0)  # Add batch dimension
            predictions, recons = self.forward(window_data)
            all_predictions.append(predictions)
            all_recons.append(recons)
        return torch.cat(all_predictions, dim=0), torch.cat(all_recons, dim=0)

# Example usage:
# model = MTAD_GAT(n_features=55, window_size=100, out_dim=55)
# time_series_data = torch.randn(1000, 55)  # Example time series data
# predictions, recons = model.sliding_window_inference(time_series_data)
