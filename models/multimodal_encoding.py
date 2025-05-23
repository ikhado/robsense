import torch.nn as nn
import torch
from einops import rearrange

from core.vit_collections import ModelConfig, TransformerConfig
from models.model_utils import TransformerLayer


class MultimodalEncoding(nn.Module):
    def __init__(self, num_modalities, model_config: ModelConfig):
        """
        Constructor for the MultimodalEncoding class.
        """
        super(MultimodalEncoding, self).__init__()

        self.num_modalities = num_modalities
        self.encoder_num_layers = model_config.transformer_config.num_layers

        self.fusion_layer = nn.Linear(self.num_modalities * model_config.hidden_size, model_config.hidden_size)

        self.layers = nn.ModuleList()

        for _ in range(self.encoder_num_layers):
            layer = TransformerLayer(transformer_config=model_config.transformer_config)
            self.layers.append(layer)
        self.encoder_norm = nn.LayerNorm(model_config.hidden_size, eps=1e-6)

    def forward(self, encoded_multimodal):
        encoded_MS, encoded_SAR = encoded_multimodal
        b, t, l, _ = encoded_MS.shape

        # Concatenate embeddings along the feature dimension
        concatenated_embeddings = torch.cat((encoded_MS, encoded_SAR), dim=-1)

        fused_embeddings = self.fusion_layer(concatenated_embeddings)

        fused_embeddings = rearrange(fused_embeddings, 'b t l d-> b (t l) d', b=b, t=t, l=l)
        x = fused_embeddings
        for layer in self.layers:
            x = layer(x)

        return self.encoder_norm(x)


class LatentReconstruction(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(LatentReconstruction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim)
        )

    def forward(self, input, mask=None):
        if mask is not None:
            input = input * mask.unsqueeze(-1)
        return self.model(input)


class VitLatentReconstruction(nn.Module):
    def __init__(self, transformer_config: TransformerConfig):
        super(VitLatentReconstruction, self).__init__()

        self.layers = nn.ModuleList()

        for _ in range(int(transformer_config.num_layers / 3)):
            layer = TransformerLayer(transformer_config=transformer_config)
            self.layers.append(layer)
        self.encoder_bnorm = nn.BatchNorm1d(transformer_config.embedding_dim)
        self.projector = nn.Linear(transformer_config.embedding_dim, transformer_config.embedding_dim)

    def forward(self, x):
        b, t, l, d = x.shape

        x = rearrange(x, 'b t l d -> b (t l) d', b=b, t=t, l=l, d=d)
        for layer in self.layers:
            x = layer(x)
        x = rearrange(x, 'b (t l) d -> b t l d', b=b, t=t, l=l, d=d)

        return x
