"""
Image Modality Encoder
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

from .layers import PositionalEncoding, patchify


class ImageModalityEncoder(nn.Module):
    """
    Image Modality Encoder with patch-level masking (CrossMAE style)

    Architecture:
        1. Patchify: [B, T, H, W] -> [B, T, num_patches, patch_dim]
        2. Filter valid land patches (94 out of 522)
        3. Remove masked patches (encoder never sees masked data)
        4. Add spatial + temporal position embeddings
        5. Transformer encoder (standard, no FiLM)
        6. Normalize (NO POOLING - keeps sequence for CrossMAE decoder)
        7. Output sequence of visible tokens [B, L_visible, d_model]

    Args:
        patch_size: Size of each patch (default: 10)
        image_hw: (H, W) image size (default: (290, 180))
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        max_time_steps: Maximum sequence length
        dropout: Dropout rate
        valid_patch_indices: Indices of valid land patches (94 patches)
    """

    def __init__(
        self,
        patch_size: int = 10,
        image_hw: Tuple[int, int] = (290, 180),
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_time_steps: int = 90,
        dropout: float = 0.1,
        valid_patch_indices: Tensor = None,
        use_weighted_fm: bool = False,  # NEW: Phase 2
        use_fm_layers: list = None,    # NEW: Which layers to save
        use_input: bool = False,        # NEW: Include input as layer 0
        modality_token: nn.Parameter = None,  # NEW: Modality token for cross-modal fusion
    ):
        super().__init__()

        self.patch_size = patch_size
        self.image_hw = image_hw
        self.d_model = d_model
        self.use_weighted_fm = use_weighted_fm
        self.use_input = use_input
        self.modality_token = modality_token  # Store modality token reference

        H, W = image_hw
        self.num_patches_h = H // patch_size  # 29
        self.num_patches_w = W // patch_size  # 18
        self.num_patches = self.num_patches_h * self.num_patches_w  # 522
        self.patch_dim = patch_size * patch_size  # 100

        # Valid patch indices (land patches only)
        if valid_patch_indices is not None:
            self.register_buffer('valid_patch_indices', valid_patch_indices)
            self.num_valid_patches = len(valid_patch_indices)
        else:
            # If not provided, use all patches
            self.register_buffer(
                'valid_patch_indices',
                torch.arange(self.num_patches, dtype=torch.long)
            )
            self.num_valid_patches = self.num_patches

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, d_model)

        # Spatial positional embedding (learnable, shared across time)
        # Only for valid patches
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, self.num_valid_patches, d_model)
        )
        nn.init.normal_(self.spatial_pos, std=0.02)

        # Temporal positional encoding (fixed sincos)
        self.temporal_pos = PositionalEncoding(d_model, max_time_steps)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm for output
        self.norm = nn.LayerNorm(d_model)

        # Phase 2: Determine which layers to save for WeightedFeatureMaps
        if use_weighted_fm:
            if use_fm_layers is None:
                # Use all layers
                self.use_fm_layers = list(range(num_layers))
            else:
                # Use specified layers
                self.use_fm_layers = use_fm_layers
        else:
            self.use_fm_layers = []

    def forward(self, x_img: Tensor, patch_mask: Tensor) -> Tuple[Tensor, Dict]:
        """
        Forward pass

        Args:
            x_img: [B, T, H, W] image sequence
            patch_mask: [B, T, 522] bool mask (True = masked, False = visible)

        Returns:
            encoder_output: [B, L_visible, d_model] - sequence of visible tokens (CrossMAE style)
            mask_info: dict with mask information for decoder
        """
        B, T, H, W = x_img.shape

        # ===== Step 1: Patchify =====
        patches = patchify(x_img, self.patch_size)  # [B, T, 522, 100]

        # ===== Step 2: Filter valid patches only =====
        patches = patches[:, :, self.valid_patch_indices, :]  # [B, T, num_valid, 100]
        patch_mask_valid = patch_mask[:, :, self.valid_patch_indices]  # [B, T, num_valid]

        # ===== Step 3: VECTORIZED selection of visible patches (NO LOOPS!) =====
        # Get visibility mask (True = visible)
        visible_mask = ~patch_mask_valid  # [B, T, num_valid]

        # Select visible patches using boolean indexing
        # This flattens the selected patches into [Total_Visible, patch_dim]
        # PyTorch iterates in row-major order (B, T, N), so order is preserved
        x_visible = patches[visible_mask]  # [Total_Visible, patch_dim]

        # Calculate number of visible patches per sample
        num_visible_per_sample = visible_mask.sum(dim=(1, 2))  # [B]
        max_len = num_visible_per_sample.max().item()

        # Edge case: no visible patches
        if max_len == 0:
            encoder_output = torch.zeros(B, 1, self.d_model, device=x_img.device, dtype=self.patch_embed.weight.dtype)
            padding_mask = torch.ones(B, 1, device=x_img.device, dtype=torch.bool)
            mask_info = {
                'mask': patch_mask,
                'lengths': [0] * B,
                'padding_mask': padding_mask,
            }
            return encoder_output, mask_info

        # Check if all samples have same length (should be true with fixed mask ratio)
        if (num_visible_per_sample == max_len).all():
            # FAST PATH: All samples have same length, no padding needed!
            x = x_visible.view(B, max_len, self.patch_dim)
            padding_mask = torch.zeros(B, max_len, device=x_img.device, dtype=torch.bool)
            lengths = [max_len] * B
        else:
            # SLOW PATH: Different lengths, need padding (shouldn't happen with fixed mask ratio)
            x = torch.zeros(B, max_len, self.patch_dim, device=x_img.device, dtype=patches.dtype)
            padding_mask = torch.zeros(B, max_len, device=x_img.device, dtype=torch.bool)
            lengths = num_visible_per_sample.cpu().tolist()

            # Fill in the data
            offset = 0
            for b in range(B):
                length = lengths[b]
                x[b, :length] = x_visible[offset:offset+length]
                if length < max_len:
                    padding_mask[b, length:] = True
                offset += length

        # ===== Step 4: Patch embedding =====
        x = self.patch_embed(x)  # [B, max_len, d_model]

        # ===== Step 5: VECTORIZED position embeddings (NO LOOPS!) =====
        num_valid = self.valid_patch_indices.shape[0]

        # Create grids of indices [B, T, num_valid]
        # Temporal indices: [[0,0...], [1,1...], ...] repeated for B
        t_indices = torch.arange(T, device=x_img.device).view(1, T, 1).expand(B, T, num_valid)

        # Spatial indices: [[0, 1, 2...], [0, 1, 2...]] repeated for B and T
        # Note: These are indices into the valid_patches array (0..93), not original 0..521
        s_indices = torch.arange(num_valid, device=x_img.device).view(1, 1, num_valid).expand(B, T, num_valid)

        # Select indices for visible patches
        t_visible = t_indices[visible_mask].view(B, -1)  # [B, max_len]
        s_visible = s_indices[visible_mask].view(B, -1)  # [B, max_len]

        # Gather spatial PE: self.spatial_pos is [1, num_valid, d_model]
        # We need to select from second dimension using s_visible
        # spatial_emb: [B, max_len, d_model]
        spatial_emb = self.spatial_pos[0, s_visible.view(-1)].view(B, max_len, -1)

        # Gather temporal PE: self.temporal_pos.pe is [1, max_time, d_model]
        # We need to select from second dimension using t_visible
        # temporal_emb: [B, max_len, d_model]
        temporal_pe = self.temporal_pos.pe.squeeze(0)  # [max_time, d_model]
        temporal_emb = temporal_pe[t_visible.view(-1)].view(B, max_len, -1)

        # Add both PEs to x (vectorized!)
        x = x + spatial_emb + temporal_emb

        # Add modality token (CAV-MAE style: after pos_embed)
        if self.modality_token is not None:
            x = x + self.modality_token  # [1, 1, d_model] broadcast to [B, max_len, d_model]

        # ===== Step 6: Transformer encoder =====
        if self.use_weighted_fm:
            # Phase 2: Collect multi-layer features
            x_feats = []

            # Optional: Include input as layer 0
            if self.use_input:
                x_feats.append(self.norm(x.clone()))

            # Process through transformer layers
            for idx, layer in enumerate(self.transformer.layers):
                x = layer(x, src_key_padding_mask=padding_mask)

                # Save specified layers
                if idx in self.use_fm_layers:
                    x_feats.append(self.norm(x.clone()))

            # Return list of features
            mask_info = {
                'mask': patch_mask,
                'lengths': lengths,
                'padding_mask': padding_mask,
            }

            return x_feats, mask_info  # List of [B, L_visible, d_model]

        else:
            # Standard: Single layer output
            x = self.transformer(x, src_key_padding_mask=padding_mask)

            # ===== Step 7: Normalize (NO POOLING - CrossMAE style) =====
            x = self.norm(x)  # [B, max_len, d_model]

            # ===== Prepare mask_info for decoder =====
            mask_info = {
                'mask': patch_mask,
                'lengths': lengths,
                'padding_mask': padding_mask,
            }

            return x, mask_info  # [B, L_visible, d_model]


if __name__ == '__main__':
    """Unit test for ImageModalityEncoder"""

    print("=" * 60)
    print("Testing ImageModalityEncoder")
    print("=" * 60)

    # Simulate valid patch indices (94 land patches)
    num_valid = 94
    valid_patch_indices = torch.randperm(522)[:num_valid].sort()[0]

    # Create encoder
    encoder = ImageModalityEncoder(
        patch_size=10,
        image_hw=(290, 180),
        d_model=256,
        nhead=8,
        num_layers=6,
        max_time_steps=90,
        dropout=0.1,
        valid_patch_indices=valid_patch_indices,
    )

    # Test data (use smaller size to avoid OOM)
    B, T, H, W = 2, 10, 290, 180  # Reduced T from 90 to 10
    x_img = torch.randn(B, T, H, W)

    # Create mask (75% of valid patches masked)
    patch_mask_full = torch.zeros(B, T, 522, dtype=torch.bool)
    for b in range(B):
        for t in range(T):
            # Randomly mask 75% of valid patches
            num_to_mask = int(num_valid * 0.75)
            masked_valid = torch.randperm(num_valid)[:num_to_mask]
            masked_patch_indices = valid_patch_indices[masked_valid]
            patch_mask_full[b, t, masked_patch_indices] = True

    print(f"Input shape: {x_img.shape}")
    print(f"Mask shape: {patch_mask_full.shape}")
    print(f"Valid patches: {num_valid}")
    print(f"Mask ratio (valid patches): {patch_mask_full[:, :, valid_patch_indices].float().mean().item():.2%}")

    # Forward pass
    token, mask_info = encoder(x_img, patch_mask_full)

    print(f"\n✓ Output token shape: {token.shape}")
    print(f"✓ Mask info keys: {mask_info.keys()}")
    print(f"✓ Visible patch counts: {mask_info['lengths']}")

    # Test backward
    loss = token.sum()
    loss.backward()

    print(f"✓ Backward pass successful")

    # Test with different mask ratios
    print(f"\n" + "=" * 60)
    print("Testing with different mask ratios")
    print("=" * 60)

    for mask_ratio in [0.0, 0.5, 0.75, 0.9]:
        patch_mask_full = torch.zeros(B, T, 522, dtype=torch.bool)
        for b in range(B):
            for t in range(T):
                num_to_mask = int(num_valid * mask_ratio)
                if num_to_mask > 0:
                    masked_valid = torch.randperm(num_valid)[:num_to_mask]
                    masked_patch_indices = valid_patch_indices[masked_valid]
                    patch_mask_full[b, t, masked_patch_indices] = True

        token, mask_info = encoder(x_img, patch_mask_full)
        visible_count = sum(mask_info['lengths'])
        expected = B * T * num_valid * (1 - mask_ratio)
        print(f"Mask ratio {mask_ratio:.0%}: "
              f"Total visible patches {visible_count} "
              f"(expected ~{expected:.0f})")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
