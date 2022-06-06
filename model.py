from random import sample
from torch import nn, Tensor
import torch
import math
import torch.nn.functional as F
from transformer import Transformer


class DETR(nn.Module):
    """
    This is the main model that performs segments detection and classification

    Copy-paste from DETR module with modifications
    """

    def __init__(self, transformer, num_classes, num_queries):
        """Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                        DETR can detect in an subsequence.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.segment_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)

    def forward(self, sample: Tensor):
        """
        Parameters:
            -- sample: batched sequences, of shape [batch_size x seq_len x embedding_len ]
                 eg, when choosing one hot encoding, embedding_len will be 5, [A, T, C, G, N]
        Return values
            -- pred_logits: the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
            -- pred_boundaries: The normalized boundaries coordinates for all queries, represented as
                            (center, width). These values are normalized in [0, 1].
        """
        pos = self.pe(sample.permute(1, 0, 2)).permute(1, 0, 2)
        hs = self.transformer(sample, self.query_embed.weight, pos)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.segment_embed(hs).sigmoid()
        out = {"pred_logits": outputs_class, "pred_boundaries": outputs_coord}
        return out


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        len = d_model // 2

        cos_pos = torch.cos(position * div_term)
        cos_pos = cos_pos[:, :len]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = cos_pos
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        output = self.pe.repeat(batch_size, 1, 1)
        return output[:, :seq_len, :]


if __name__ == "__main__":
    n, s, e = 10, 100, 5
    num_queries = 100
    transformer = Transformer(d_model=5, nhead=5)
    model = DETR(transformer=transformer, num_classes=11, num_queries=num_queries)
    x = torch.rand(n, s, e)
    print(x.shape)
    output = model(x)
    print(output["pred_logits"].shape, output["pred_boundaries"].shape)
