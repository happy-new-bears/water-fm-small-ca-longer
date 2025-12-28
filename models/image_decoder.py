"""
Image Modality Decoder (CrossMAE style)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

from .layers import PositionalEncoding, CrossAttentionBlock, WeightedFeatureMaps


class ImageModalityDecoder(nn.Module):
    """
    Image Modality Decoder for MAE reconstruction (CrossMAE Architecture)

    Architecture (Phase 1 - Complete CrossMAE):
        1. Accept encoder sequence [B, L_visible, d_model]
        2. Create masked queries (only for masked positions)
        3. Add spatial + temporal position embeddings to queries
        4. CrossAttention decoder: queries attend to encoder sequence
        5. Linear head to predict patch values

    Phase 2 Enhancement (WeightedFeatureMaps):
        - Accept list of encoder features from multiple layers
        - Learn weighted combinations for each decoder layer
        - Each decoder layer uses a different weighted feature map

    This is the CORE of CrossMAE speedup:
    - Standard MAE: All positions (visible + masked) do self-attention O(N²)
    - CrossMAE: Only masked positions as queries, attend to visible O(M×N)
    - Speedup: ~3-4x when mask_ratio=0.75

    Args:
        encoder_dim: Encoder output dimension
        decoder_dim: Decoder embedding dimension
        num_patches: Total number of patches (522)
        patch_dim: Patch dimension (100 for 10x10 patches)
        num_decoder_layers: Number of transformer layers
        nhead: Number of attention heads
        max_time_steps: Maximum sequence length
        dropout: Dropout rate
        use_cross_attn: Use CrossAttention (default: True)
        decoder_self_attn: Use self-attention in decoder (default: False)
        use_weighted_fm: Use WeightedFeatureMaps (Phase 2, default: False)
        num_encoder_layers: Number of encoder layers for WeightedFeatureMaps
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        decoder_dim: int = 128,
        num_patches: int = 522,
        patch_dim: int = 100,
        num_decoder_layers: int = 4,
        nhead: int = 8,
        max_time_steps: int = 90,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
        decoder_self_attn: bool = False,
        use_weighted_fm: bool = False,  # NEW: Phase 2
        num_encoder_layers: int = 6,    # NEW: Phase 2
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim
        self.max_time_steps = max_time_steps
        self.use_cross_attn = use_cross_attn
        self.use_weighted_fm = use_weighted_fm  # NEW: Phase 2
        self.num_decoder_layers = num_decoder_layers  # NEW: Save for later

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Spatial positional embedding (learnable, for all 522 patches)
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, num_patches, decoder_dim)
        )
        nn.init.normal_(self.spatial_pos, std=0.02)

        # Temporal positional encoding (fixed sincos)
        self.temporal_pos = PositionalEncoding(decoder_dim, max_time_steps)

        # Decoder blocks
        if use_cross_attn:
            # CrossAttention decoder (CrossMAE)
            self.decoder_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    encoder_dim=encoder_dim,
                    decoder_dim=decoder_dim,
                    num_heads=nhead,
                    mlp_ratio=4.0,
                    drop=dropout,
                    attn_drop=dropout,
                    self_attn=decoder_self_attn,
                )
                for _ in range(num_decoder_layers)
            ])
            self.decoder_norm = nn.LayerNorm(decoder_dim)

            # Phase 2: WeightedFeatureMaps
            if use_weighted_fm:
                # WeightedFeatureMaps module
                self.weighted_fm = WeightedFeatureMaps(
                    num_layers=num_encoder_layers,
                    embed_dim=encoder_dim,
                    decoder_depth=num_decoder_layers,
                )

                # Layer-wise normalization (one for each decoder layer)
                self.dec_norms = nn.ModuleList([
                    nn.LayerNorm(encoder_dim)
                    for _ in range(num_decoder_layers)
                ])
        else:
            # Fallback: Self-attention decoder (standard MAE)
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim,
                nhead=nhead,
                dim_feedforward=4 * decoder_dim,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                decoder_layer,
                num_layers=num_decoder_layers,
            )
            self.decoder_norm = nn.LayerNorm(decoder_dim)

        # Prediction head
        self.pred_head = nn.Linear(decoder_dim, patch_dim)

    def forward(
        self,
        encoder_output,
        mask_info: Dict,
        decoder_modality_token=None,
        encoder_padding_mask=None  # ⭐ NEW: [B, L_total] global padding mask
    ) -> Tensor:
        """
        Forward pass

        Args:
            encoder_output: Encoder features
                - If use_weighted_fm=False: [B, L_visible, encoder_dim]
                - If use_weighted_fm=True: list of [B, L_visible, encoder_dim]
            mask_info: dict with 'mask' [B, T, num_patches], 'padding_mask' [B, L_visible]
            decoder_modality_token: [1, 1, decoder_dim] decoder modality token (optional)
            encoder_padding_mask: [B, L_total] global padding mask for cross-attention (optional)

        Returns:
            pred_patches: [B, T, num_patches, patch_dim] - predicted patches
        """
        if self.use_cross_attn:
            return self._forward_cross_attn(
                encoder_output, mask_info, decoder_modality_token, encoder_padding_mask
            )
        else:
            return self._forward_self_attn(
                encoder_output, mask_info, decoder_modality_token, encoder_padding_mask
            )

    def _forward_cross_attn(
        self,
        encoder_output,
        mask_info: Dict,
        decoder_modality_token=None,
        encoder_padding_mask=None  # ⭐ NEW: [B, L_total] global padding mask
    ) -> Tensor:
        """
        Vectorized CrossMAE decoder (NO LOOPS over batch!)

        Key optimization: Process entire batch in parallel instead of per-sample loops.
        Assumes fixed mask ratio, so all samples have same number of masked patches.
        """
        mask = mask_info['mask']  # [B, T, num_patches]
        padding_mask = mask_info.get('padding_mask')  # [B, L_visible]
        B, T, num_patches = mask.shape

        # ===== Phase 2: Process encoder features =====
        if self.use_weighted_fm:
            # encoder_output is list of [B, L_visible, encoder_dim]
            assert isinstance(encoder_output, list), "Expected list of encoder features"
            # Combine multi-layer features: list -> [B, L, C, decoder_depth]
            weighted_features = self.weighted_fm(encoder_output)  # [B, L_visible, encoder_dim, num_decoder_layers]
            encoder_features_per_layer = weighted_features
        else:
            # Standard: single encoder output [B, L_visible, encoder_dim]
            encoder_features_per_layer = None

        # ===== Step 1: VECTORIZED Query Creation (NO LOOPS!) =====
        # Assumption: Fixed mask ratio means each sample has same number of masked patches
        num_masked_total = mask.sum().item()
        num_masked_per_sample = num_masked_total // B

        # Get all masked positions
        # nonzero() returns indices in order (b, t, p), which we rely on
        indices = mask.nonzero(as_tuple=False)  # [Total_Masked, 3]

        # Extract t and p indices for Position Embedding
        t_indices = indices[:, 1].view(B, num_masked_per_sample)  # [B, k]
        p_indices = indices[:, 2].view(B, num_masked_per_sample)  # [B, k]

        # Create Queries [B, k, decoder_dim]
        queries = self.mask_token.expand(B, num_masked_per_sample, -1).clone()

        # Add Spatial PE (Gathering, NO LOOP!)
        # self.spatial_pos: [1, num_patches, decoder_dim]
        # p_indices: [B, k] -> Gather -> [B, k, decoder_dim]
        spatial_emb = self.spatial_pos[0, p_indices]  # [B, k, decoder_dim]
        queries = queries + spatial_emb

        # Add Temporal PE (Gathering, NO LOOP!)
        # temporal_pos.pe: [1, max_len, decoder_dim] -> [max_len, decoder_dim]
        temporal_emb = self.temporal_pos.pe.squeeze(0)[t_indices]  # [B, k, decoder_dim]
        queries = queries + temporal_emb

        # Add Decoder Modality Token (CAV-MAE style: after pos_embed)
        if decoder_modality_token is not None:
            queries = queries + decoder_modality_token  # [1, 1, decoder_dim] broadcast to [B, k, decoder_dim]

        # ===== Step 2: Batched Cross Attention (PARALLEL!) =====
        x = queries  # [B, k, decoder_dim]

        for layer_idx, blk in enumerate(self.decoder_blocks):
            # Get Encoder Features
            if self.use_weighted_fm:
                # [B, L_visible, encoder_dim, Layers] -> [B, L_visible, encoder_dim]
                batch_encoder = encoder_features_per_layer[:, :, :, layer_idx]

                # Apply Layer-wise Norm
                batch_encoder = self.dec_norms[layer_idx](batch_encoder)
            else:
                # Standard
                if isinstance(encoder_output, list):
                    batch_encoder = encoder_output[-1]  # Use last layer
                else:
                    batch_encoder = encoder_output  # [B, L_visible, encoder_dim]

            # CrossAttention now processes ENTIRE BATCH in parallel!
            # ⭐ NEW: Pass encoder_padding_mask to cross-attention
            x = blk(x, batch_encoder, memory_key_padding_mask=encoder_padding_mask)  # [B, k, decoder_dim]

        # ===== Step 3: Prediction =====
        x = self.decoder_norm(x)  # [B, k, decoder_dim]
        predictions = self.pred_head(x)  # [B, k, patch_dim]

        # ===== Step 4: Scatter back to full image (VECTORIZED!) =====
        # Use boolean indexing to directly assign, no loops!
        dtype = predictions.dtype
        device = predictions.device
        pred_patches = torch.zeros(B, T, num_patches, self.patch_dim, device=device, dtype=dtype)

        # predictions.reshape(-1, patch_dim) -> [Total_Masked, patch_dim]
        # mask selects exactly [Total_Masked] positions
        # PyTorch guarantees nonzero() order matches the flattened order
        pred_patches[mask] = predictions.reshape(-1, self.patch_dim)

        return pred_patches

    def _forward_self_attn(
        self,
        encoder_output: Tensor,
        mask_info: Dict,
        decoder_modality_token=None,
        encoder_padding_mask=None  # ⭐ NEW: For interface consistency (not used in self-attn mode)
    ) -> Tensor:
        """
        Fallback: Standard self-attention decoder (for compatibility)
        """
        mask = mask_info['mask']  # [B, T, num_patches]
        padding_mask = mask_info.get('padding_mask')  # [B, L_visible]
        B, T, num_patches = mask.shape

        # Pool encoder sequence to single token
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()  # [B, L_visible, 1]
            encoder_token = (encoder_output * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        else:
            encoder_token = encoder_output.mean(dim=1)  # [B, encoder_dim]

        # Create full sequence with mask tokens
        x = self.mask_token.expand(B, T, num_patches, -1).clone()  # [B, T, num_patches, decoder_dim]

        # Fill visible positions with pooled encoder token
        for b in range(B):
            visible_mask = ~mask[b]  # [T, num_patches]
            if visible_mask.any():
                x[b][visible_mask] = encoder_token[b]

        # Add position embeddings
        x = x + self.spatial_pos  # Broadcast spatial PE

        # Add temporal PE
        B_orig, T_orig, P, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B_orig * P, T_orig, D)
        x = self.temporal_pos(x)
        x = x.reshape(B_orig, P, T_orig, D).permute(0, 2, 1, 3).contiguous()

        # Add Decoder Modality Token (if provided)
        if decoder_modality_token is not None:
            x = x + decoder_modality_token  # [1, 1, decoder_dim] broadcast

        # Flatten for transformer
        x = x.reshape(B, T * num_patches, self.decoder_dim)

        # Self-attention transformer
        x = self.transformer(x)
        x = self.decoder_norm(x)

        # Reshape and predict
        x = x.reshape(B, T, num_patches, self.decoder_dim)
        pred_patches = self.pred_head(x)

        return pred_patches


if __name__ == '__main__':
    """Unit test for ImageModalityDecoder"""

    print("=" * 60)
    print("Testing ImageModalityDecoder")
    print("=" * 60)

    # Create decoder
    decoder = ImageModalityDecoder(
        encoder_dim=256,
        decoder_dim=128,
        num_patches=522,
        patch_dim=100,
        num_decoder_layers=4,
        nhead=8,
        max_time_steps=90,
        dropout=0.1,
    )

    # Test data (use smaller T to avoid OOM)
    B, T = 2, 10
    encoder_token = torch.randn(B, 256)  # [B, encoder_dim]

    # Create mask (75% masked)
    mask = torch.rand(B, T, 522) < 0.75  # [B, T, 522] bool

    mask_info = {'mask': mask}

    print(f"Encoder token shape: {encoder_token.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.float().mean().item():.2%}")

    # Forward pass
    pred_patches = decoder(encoder_token, mask_info)

    print(f"\n✓ Predicted patches shape: {pred_patches.shape}")
    assert pred_patches.shape == (B, T, 522, 100), \
        f"Expected shape ({B}, {T}, 522, 100), got {pred_patches.shape}"

    # Test backward
    loss = pred_patches.sum()
    loss.backward()

    print(f"✓ Backward pass successful")

    # Test reconstruction loss
    target_patches = torch.randn(B, T, 522, 100)
    mse_loss = torch.nn.functional.mse_loss(
        pred_patches, target_patches, reduction='none'
    )  # [B, T, 522, 100]

    # Average over patch dimension
    mse_loss = mse_loss.mean(dim=-1)  # [B, T, 522]

    # Only compute loss on masked patches
    masked_loss = (mse_loss * mask.float()).sum() / mask.sum()

    print(f"✓ Masked reconstruction loss: {masked_loss.item():.4f}")

    # Test with different mask ratios
    print(f"\n" + "=" * 60)
    print("Testing with different mask ratios")
    print("=" * 60)

    for mask_ratio in [0.0, 0.5, 0.75]:  # Reduced to avoid OOM
        mask = torch.rand(B, T, 522) < mask_ratio
        mask_info = {'mask': mask}
        pred_patches = decoder(encoder_token, mask_info)
        print(f"Mask ratio {mask_ratio:.0%}: "
              f"Output shape {pred_patches.shape}, "
              f"Num masked: {mask.sum().item()}/{B*T*522}")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
