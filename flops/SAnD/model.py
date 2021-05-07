import torch
import torch.nn as nn
from flops.SAnD import modules
from thop import profile

class EncoderLayerForSAnD(nn.Module):
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=128, dropout_rate=0.2) -> None:
        super(EncoderLayerForSAnD, self).__init__()
        self.d_model = d_model

        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.positional_encoding = modules.PositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)

        return x


class SAnD(nn.Module):
    """
    Simply Attend and Diagnose model
    The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18)
    `Attend and Diagnose: Clinical Time Series Analysis Using Attention Models <https://arxiv.org/abs/1711.03905>`_
    Huan Song, Deepta Rajan, Jayaraman J. Thiagarajan, Andreas Spanias
    """
    def __init__(
            self, input_features: int, seq_len: int, n_heads: int, factor: int,
            n_class: int, n_layers: int, d_model: int = 128, dropout_rate: float = 0.2
    ) -> None:
        super(SAnD, self).__init__()
        self.encoder = EncoderLayerForSAnD(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate)
        self.dense_interpolation = modules.DenseInterpolation(seq_len, factor)
        self.clf = modules.ClassificationModule(d_model, factor, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.dense_interpolation(x)
        x = self.clf(x)
        return x

if __name__=='__main__':

    print('Constructing model ... ')
    # device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
    device = 'cpu'
    print("available device: {}".format(device))
    batch_x = torch.rand(128, 400, 76)
    batch_x = torch.tensor(batch_x, dtype=torch.float32).to(device)

    in_feature = 76
    seq_len = 400
    n_heads = 32
    factor = 32
    num_class = 2
    num_layers = 6
    model = SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers)
    model = model.to(device)

    # cur_output = model(batch_x)
    flops, params = profile(model, inputs=[batch_x])
    print("%.2fG" % (flops / 1e9), "%.2fM" % (params / 1e6))
    print('!!!!!!!')