import torch
from torch import nn
from gat_layers import ConvLayer, TemporalAttentionLayer, FeatureAttentionLayer, Forecasting_Model, GRULayer


class ConstantGuesser(nn.Module):

    def __init__(self, constant, output_steps):
        super().__init__()
        self.constant = constant
        self.output_steps = output_steps

    def forward(self, x):
        return torch.full((x.shape[0], self.output_steps, 1), self.constant)


class NaivePropagator(nn.Module):
    def __init__(self, output_steps) -> None:
        super().__init__()
        self.output_steps = output_steps

    def forward(self, x):
        batch_size = x.shape[0]
        out = x[:, -1, 0].reshape(batch_size, 1, 1)
        return torch.repeat_interleave(out, repeats=self.output_steps, dim=1)


class Linear(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x).unsqueeze(dim=2)


class EncoderDecoder(nn.Module):
    def __init__(self, enc_in_features, dec_in_features, hidden_size, out_features, num_layers=12, enc_dropout=0, dec_dropout=0):
        super().__init__()
        self.encoder = nn.GRU(input_size=enc_in_features, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dropout=enc_dropout)
        self.decoder = nn.GRU(input_size=dec_in_features, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True, dropout=dec_dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=out_features)

    def forward(self, x):
        past, future = x
        _, enc_hidden = self.encoder(past)
        out, _ = self.decoder(future, enc_hidden)
        return self.linear(out)


class GATModel(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        input_steps,
        num_features,
        output_steps,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        dropout=0.2,
        alpha=0.2,
    ):
        super().__init__()
        self.conv = ConvLayer(num_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            num_features, input_steps, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(
            num_features, input_steps, dropout, alpha, time_gat_embed_dim, use_gatv2)
        self.gru = GRULayer(3 * num_features, gru_hid_dim, gru_n_layers, dropout)
        self.forecasting_model = Forecasting_Model(
            gru_hid_dim, forecast_hid_dim, output_steps, forecast_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)
        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)
        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp
        predictions = self.forecasting_model(h_end).unsqueeze(dim=2)
        return predictions
