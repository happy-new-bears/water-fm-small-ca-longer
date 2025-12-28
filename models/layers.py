"""
Shared components for Multi-modal MAE
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding

    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length (default: 90 for temporal encoding)
    """
    def __init__(self, d_model: int, max_len: int = 90):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, d_model] or [B, N, T, d_model]

        Returns:
            x + pe: Same shape as input
        """
        return x + self.pe[:, : x.size(1), :]


class FiLMLayerNorm(nn.Module):
    """
    Feature-wise Linear Modulation with LayerNorm

    Formula: gamma * LayerNorm(x) + beta

    Args:
        d_model: Feature dimension
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, d_model] or [B, N, T, d_model]
            gamma: [B, 1, d_model] modulation scale
            beta: [B, 1, d_model] modulation shift

        Returns:
            Modulated features: Same shape as x
        """
        return gamma * self.ln(x) + beta


class FiLMEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with FiLM modulation

    Structure:
        x = FiLM(x + MultiheadAttention(x), gamma, beta)
        x = FiLM(x + FFN(x), gamma, beta)

    Args:
        d_model: Feature dimension
        nhead: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.film1 = FiLMLayerNorm(d_model)
        self.film2 = FiLMLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        gamma: Tensor,
        beta: Tensor,
        key_padding_mask: Tensor = None
    ) -> Tensor:
        """
        Args:
            x: [B, T, d_model] input sequence
            gamma: [B, 1, d_model] modulation scale
            beta: [B, 1, d_model] modulation shift
            key_padding_mask: [B, T] bool mask (True = padding)

        Returns:
            Output: [B, T, d_model]
        """
        # Self-attention with residual + FiLM
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.film1(x + attn_out, gamma, beta)

        # FFN with residual + FiLM
        x = self.film2(x + self.ffn(x), gamma, beta)

        return x


def patchify(x_img: Tensor, patch_size: int = 10) -> Tensor:
    """
    Convert image to patches

    Args:
        x_img: [B, T, H, W] image tensor (H=290, W=180)
        patch_size: Size of each patch (default: 10)

    Returns:
        patches: [B, T, num_patches, patch_dim]
                 num_patches = (H//patch_size) * (W//patch_size) = 29*18 = 522
                 patch_dim = patch_size * patch_size = 100
    """
    B, T, H, W = x_img.shape

    # Calculate number of patches
    num_patches_h = H // patch_size  # 290 // 10 = 29
    num_patches_w = W // patch_size  # 180 // 10 = 18

    # Reshape to patches
    # [B, T, H, W] -> [B, T, num_patches_h, patch_size, num_patches_w, patch_size]
    x = x_img.reshape(B, T, num_patches_h, patch_size, num_patches_w, patch_size)

    # Permute to [B, T, num_patches_h, num_patches_w, patch_size, patch_size]
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Flatten patches: [B, T, num_patches, patch_dim]
    patches = x.reshape(B, T, num_patches_h * num_patches_w, patch_size * patch_size)

    return patches


def unpatchify(patches: Tensor, patch_size: int = 10, image_hw: tuple = (290, 180)) -> Tensor:
    """
    Convert patches back to image

    Args:
        patches: [B, T, num_patches, patch_dim] patch tensor
        patch_size: Size of each patch (default: 10)
        image_hw: (H, W) original image size (default: (290, 180))

    Returns:
        x_img: [B, T, H, W] reconstructed image
    """
    B, T, num_patches, patch_dim = patches.shape
    H, W = image_hw

    num_patches_h = H // patch_size  # 29
    num_patches_w = W // patch_size  # 18

    # Reshape patches to [B, T, num_patches_h, num_patches_w, patch_size, patch_size]
    x = patches.reshape(B, T, num_patches_h, num_patches_w, patch_size, patch_size)

    # Permute to [B, T, num_patches_h, patch_size, num_patches_w, patch_size]
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Reshape to [B, T, H, W]
    x_img = x.reshape(B, T, H, W)

    return x_img


class CrossAttention(nn.Module):
    """
    Cross-Attention module for CrossMAE decoder

    Query from decoder (x), Key and Value from encoder (y)

    Reference: CrossMAE transformer_utils.py:69-108

    Args:
        encoder_dim: Dimension of encoder features
        decoder_dim: Dimension of decoder features
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in Q, K, V projections
        qk_scale: Manual scale factor for attention (default: head_dim ** -0.5)
        attn_drop: Attention dropout rate
        proj_drop: Projection dropout rate
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Q projection (from decoder)
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)

        # K, V projection (from encoder)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        key_padding_mask: Tensor = None  # ⭐ NEW: [B, Ny] bool mask (True=padding)
    ) -> Tensor:
        """
        Forward pass

        Args:
            x: [B, N_decoder, decoder_dim] - decoder queries
            y: [B, N_encoder, encoder_dim] - encoder keys/values
            key_padding_mask: [B, N_encoder] bool mask (True=padding, optional)

        Returns:
            output: [B, N_decoder, decoder_dim] - cross-attended features
        """
        B, N, C = x.shape
        Ny = y.shape[1]

        # Query from decoder
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B, num_heads, N, head_dim]

        # Key, Value from encoder
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [2, B, num_heads, Ny, head_dim]
        k, v = kv[0], kv[1]
        # Each: [B, num_heads, Ny, head_dim]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, Ny]

        # ⭐ NEW: Apply key padding mask before softmax
        if key_padding_mask is not None:
            # key_padding_mask: [B, Ny] -> [B, 1, 1, Ny]
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Ny]
            attn = attn.masked_fill(attn_mask, float('-inf'))  # Mask padded keys

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, decoder_dim]

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttentionBlock(nn.Module):
    """
    Transformer block with cross-attention for CrossMAE decoder

    Structure:
        1. Optional self-attention (masked tokens attend to each other)
        2. Cross-attention (masked queries attend to encoder keys/values)
        3. FFN (MLP)

    Reference: CrossMAE transformer_utils.py:129-156

    Args:
        encoder_dim: Dimension of encoder features
        decoder_dim: Dimension of decoder features
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to decoder_dim (default: 4)
        qkv_bias: Whether to use bias in Q, K, V projections
        qk_scale: Manual scale factor for attention
        drop: Dropout rate
        attn_drop: Attention dropout rate
        self_attn: Whether to include self-attention before cross-attention
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: float = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        self_attn: bool = False
    ):
        super().__init__()

        self.has_self_attn = self_attn

        # Optional self-attention
        if self_attn:
            self.norm0 = nn.LayerNorm(decoder_dim)
            self.self_attn = nn.MultiheadAttention(
                decoder_dim,
                num_heads,
                dropout=attn_drop,
                batch_first=True
            )

        # Cross-attention (queries from decoder, keys/values from encoder)
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.cross_attn = CrossAttention(
            encoder_dim,
            decoder_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        # MLP
        self.norm2 = nn.LayerNorm(decoder_dim)
        mlp_hidden_dim = int(decoder_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, decoder_dim),
            nn.Dropout(drop)
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        memory_key_padding_mask: Tensor = None  # ⭐ NEW: [B, N_encoder] padding mask for encoder
    ) -> Tensor:
        """
        Forward pass

        Args:
            x: [B, N_decoder, decoder_dim] - decoder queries
            y: [B, N_encoder, encoder_dim] - encoder keys/values
            memory_key_padding_mask: [B, N_encoder] bool mask (True=padding, optional)

        Returns:
            output: [B, N_decoder, decoder_dim] - processed features
        """
        # Optional self-attention
        if self.has_self_attn:
            x_norm = self.norm0(x)
            x = x + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # Cross-attention with key padding mask
        x = x + self.cross_attn(self.norm1(x), y, key_padding_mask=memory_key_padding_mask)

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class WeightedFeatureMaps(nn.Module):
    """
    Learnable weighted combination of multi-layer encoder features

    This module learns how to combine features from different encoder layers
    for each decoder layer. Each decoder layer gets a different weighted
    combination of encoder features.

    Reference: CrossMAE models_cross.py:23-40

    Args:
        num_layers: Number of encoder layers to combine (k)
        embed_dim: Embedding dimension (not used, kept for compatibility)
        decoder_depth: Number of decoder layers (j)

    Input:
        feature_maps: List of [B, L, C] tensors from encoder layers
        Length of list = num_layers (k)

    Output:
        weighted_features: [B, L, C, decoder_depth]
        Each decoder layer j gets features[..., j]
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        decoder_depth: int
    ):
        super().__init__()

        self.num_layers = num_layers
        self.decoder_depth = decoder_depth

        # Linear layer: [k] -> [decoder_depth]
        # Learns weights for combining k encoder layers into decoder_depth combinations
        self.linear = nn.Linear(num_layers, decoder_depth, bias=False)

        # Initialize with small random weights
        std_dev = 1. / math.sqrt(num_layers)
        nn.init.normal_(self.linear.weight, mean=0., std=std_dev)

    def forward(self, feature_maps: list) -> Tensor:
        """
        Forward pass

        Args:
            feature_maps: List of [B, L, C] encoder layer outputs
                         Length = num_layers (k)

        Returns:
            output: [B, L, C, decoder_depth] weighted combinations
        """
        # Validate input
        assert isinstance(feature_maps, list), "Input should be a list of feature maps"
        assert len(feature_maps) == self.num_layers, \
            f"Expected {self.num_layers} feature maps, got {len(feature_maps)}"

        # Stack: list of [B, L, C] -> [B, L, C, k]
        stacked = torch.stack(feature_maps, dim=-1)  # [B, L, C, k]

        # Weighted combination: [B, L, C, k] -> [B, L, C, decoder_depth]
        # For each position (b, l, c), compute decoder_depth different
        # weighted combinations of the k encoder features
        output = self.linear(stacked)  # [B, L, C, decoder_depth]

        return output

