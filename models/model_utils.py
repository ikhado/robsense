import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import LayerNorm, Linear
import random

from core.vit_collections import TransformerConfig


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token, doy):
    """
    # This code is adapted from K. He et. al "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, doy)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, doy=1):
    """
    # This code is adapted from K. He et. al "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)
    omega = omega * doy

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, doy=1):
    """
    # This code is adapted from K. He et. al "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022
    @param embed_dim:
    @param grid:
    @return:
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], doy)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], doy)  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def count_parameters(model):
    """
    Count the total number of parameters in a PyTorch model.

    Args:
    model (torch.nn.Module): The PyTorch model.

    Returns:
    int: The total number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout):
        """
        Constructor for the MLP class.

        @param embed_dim: The input embedding dimension.
        @param mlp_dim: The dimension of the hidden layer in the MLP.
        @param dropout: The dropout probability.

        """
        super(MLP, self).__init__()
        self.mlp_hidden_dim, self.mlp_dim, self.dropout = embed_dim, mlp_dim, dropout
        self.fc1 = Linear(in_features=embed_dim, out_features=mlp_dim)
        self.fc2 = Linear(in_features=mlp_dim, out_features=embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout)

    def __init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        assert self.head_dim * heads == dim, "dim must be divisible by heads"

        self.query_linear = nn.Linear(dim, dim)
        self.key_linear = nn.Linear(dim, dim)
        self.value_linear = nn.Linear(dim, dim)
        self.out_linear = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # Linear projections
        q = self.query_linear(q).view(batch_size, -1, self.heads, self.head_dim)
        k = self.key_linear(k).view(batch_size, -1, self.heads, self.head_dim)
        v = self.value_linear(v).view(batch_size, -1, self.heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, heads, seq_length, head_dim)
        k = k.transpose(1, 2)  # (batch_size, heads, seq_length, head_dim)
        v = v.transpose(1, 2)  # (batch_size, heads, seq_length, head_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.softmax(dim=-1)
        output = torch.matmul(attention_scores, v)  # (batch_size, heads, seq_length, head_dim)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)

        # Final linear projection
        output = self.out_linear(output)

        return output


class TransformerLayer(nn.Module):
    def __init__(self, transformer_config: TransformerConfig):
        super(TransformerLayer, self).__init__()
        self.embed_dim, self.heads = transformer_config.embedding_dim, transformer_config.num_heads
        self.mlp_dim, self.dropout = transformer_config.mlp_dim, transformer_config.dropout_rate
        self.norm_before = LayerNorm(self.embed_dim)
        self.attn = MultiHeadAttention(self.embed_dim, self.heads)

        self.mlp_norm = LayerNorm(self.embed_dim)
        self.mlp = MLP(embed_dim=self.embed_dim, mlp_dim=self.mlp_dim, dropout=self.dropout)

    def forward(self, x):
        x_tmp = x
        x = self.norm_before(x)
        x = self.attn(x, x, x)
        x = x + x_tmp

        x_tmp = x
        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + x_tmp
        return x


def create_patches(images, patch_size):
    '''

    Create non-overlapping patches from the images.

    @param images: images.shape = (batch_size, times, channels, height, width)
    @return: output.shape = (batch_size, times, num_patches, channels, patch_height, patch_width)
    '''

    batch_size, times, channels, height, width = images.shape
    assert height % patch_size == 0 and width % patch_size == 0, "Height and width must be divisible by patch size"

    patches = images.unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, times, channels, -1, patch_size,
                                        patch_size)  # (batch_size, times, channels, num_patches, patch_size, patch_size)
    patches = patches.permute(0, 1, 3, 2, 4, 5)  # (batch_size, times, num_patches, channels, patch_size, patch_size)
    return patches


def reconstruct_images(patches, image_shape, patch_size):
    """
    Reconstruct the images from patches.
    @param patches: the patches generated from image. patches.shape = (batch_size, times, num_patches, channels, patch_size, patch_size)
    @param image_shape: shape of the original image
    """
    batch_size, times, channels, height, width = image_shape
    num_patches_per_row = height // patch_size
    num_patches_per_col = width // patch_size

    patches = patches.permute(0, 1, 3, 2, 4, 5)  # (batch_size, times, channels, num_patches, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, times, channels, num_patches_per_row, num_patches_per_col,
                                        patch_size, patch_size)
    patches = patches.permute(0, 1, 2, 3, 5, 4,
                              6)  # (batch_size, times, channels, num_patches_per_row, patch_size, num_patches_per_col, patch_size)
    _reconstruct_images = patches.contiguous().view(batch_size, times, channels, height, width)
    return _reconstruct_images


def mask_patches(patches, patch_mask_rate, masked_patch_value):
    '''
    Randomly mask a percentage of patches.
    '''
    batch_size, times, num_patches, channels, patch_size, _ = patches.shape
    num_masked = int(patch_mask_rate * num_patches)
    mask_indices = torch.zeros((batch_size, times, num_patches), dtype=torch.bool)

    for b in range(batch_size):
        for t in range(times):
            # generate random indices to mask
            mask_idx = random.sample(range(num_patches), num_masked)
            mask_indices[b, t, mask_idx] = True

    patches[mask_indices] = masked_patch_value
    return patches


def mask_images_and_channels(inputs):
    """
    Masks images and channels in the input tensor, and appends zero values to the end of each image.

    Args:
    - inputs (torch.Tensor): Input tensor of shape (batch_size, times, channels, height, width).
    - image_mask_rate (float): Rate at which to mask images.
    - channel_mask_rate (float): Rate at which to mask channels within each image.

    Returns:
    - torch.Tensor: Tensor with masked images and channels, and zero values appended.
    """
    batch_size, times, channels, height, width = inputs.shape

    image_mask_rate = random.uniform(0, 0.875)

    # Calculate the number of images to mask
    num_images_to_mask = int(image_mask_rate * times)

    channel_mask_rate = random.uniform(0, 1)

    # Calculate the number of channels to mask
    num_channels_to_mask = int(channel_mask_rate * (channels - 3))

    # Create a copy of the inputs to avoid modifying the original tensor
    masked_inputs = inputs.clone()

    for b in range(batch_size):
        # Select random time indices to mask
        time_indices_to_mask = random.sample(range(times), num_images_to_mask)
        for t in time_indices_to_mask:
            masked_inputs[b, t] = 0  # Mask entire image
        if channels == 2:
            continue
        else:
            for t in range(times):
                if t not in time_indices_to_mask:
                    # Select random channels to mask
                    channels_to_mask = random.sample(range(3, channels), num_channels_to_mask)
                    remaining_channels = [c for c in range(channels) if c not in channels_to_mask]
                    # Create new tensor with remaining channels and zeros appended
                    new_image = torch.zeros(channels, height, width, device=inputs.device)
                    new_image[:len(remaining_channels)] = masked_inputs[b, t, remaining_channels]
                    masked_inputs[b, t] = new_image

    return masked_inputs


def freeze_model(model):
    """
    Freeze all the parameters in the model so that the grad will not
    be computed in backward.
    """
    for param in model.parameters():
        param.requires_grad = False
