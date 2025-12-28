"""
Multi-modal MAE Main Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional
import os

from .image_encoder import ImageModalityEncoder
from .vector_encoder import VectorModalityEncoder
from .image_decoder import ImageModalityDecoder
from .vector_decoder import VectorModalityDecoder
from .layers import patchify, unpatchify


class MultiModalMAE(nn.Module):
    """
    Multi-modal Masked Autoencoder for Hydrology Data

    Supports 5 modalities:
        - 3 Image modalities: precipitation, soil_moisture, temperature
        - 2 Vector modalities: evaporation, riverflow

    Each modality has independent encoder and decoder.
    Vector encoders use FiLM mechanism to fuse static attributes.

    Args:
        config: MAEConfig object with model configuration
        valid_patch_indices: Tensor of valid land patch indices (94 patches)
    """

    def __init__(self, config, valid_patch_indices: Tensor = None):
        super().__init__()

        self.config = config

        # Store valid patch indices for land mask
        if valid_patch_indices is not None:
            self.register_buffer('valid_patch_indices', valid_patch_indices)
        else:
            # Default: use all patches
            num_patches = (config.image_height // config.patch_size) * \
                         (config.image_width // config.patch_size)
            self.register_buffer(
                'valid_patch_indices',
                torch.arange(num_patches, dtype=torch.long)
            )

        # ========== Modality Tokens (Encoder) ==========
        # 5个encoder modality tokens (d_model维度)
        self.modality_precip = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.modality_soil = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.modality_temp = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.modality_evap = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.modality_riverflow = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # ========== Decoder Modality Tokens ==========
        # 5个decoder modality tokens (decoder_dim维度)
        self.decoder_modality_precip = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
        self.decoder_modality_soil = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
        self.decoder_modality_temp = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
        self.decoder_modality_evap = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))
        self.decoder_modality_riverflow = nn.Parameter(torch.zeros(1, 1, config.decoder_dim))

        # ========== Image Encoders ==========
        self.precip_encoder = ImageModalityEncoder(
            patch_size=config.patch_size,
            image_hw=(config.image_height, config.image_width),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.img_encoder_layers,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            valid_patch_indices=self.valid_patch_indices,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            use_fm_layers=config.use_fm_layers,      # NEW: Phase 2
            use_input=config.use_input,              # NEW: Phase 2
            modality_token=self.modality_precip,     # NEW: Cross-modal fusion
        )

        self.soil_encoder = ImageModalityEncoder(
            patch_size=config.patch_size,
            image_hw=(config.image_height, config.image_width),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.img_encoder_layers,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            valid_patch_indices=self.valid_patch_indices,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            use_fm_layers=config.use_fm_layers,      # NEW: Phase 2
            use_input=config.use_input,              # NEW: Phase 2
            modality_token=self.modality_soil,       # NEW: Cross-modal fusion
        )

        self.temp_encoder = ImageModalityEncoder(
            patch_size=config.patch_size,
            image_hw=(config.image_height, config.image_width),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.img_encoder_layers,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            valid_patch_indices=self.valid_patch_indices,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            use_fm_layers=config.use_fm_layers,      # NEW: Phase 2
            use_input=config.use_input,              # NEW: Phase 2
            modality_token=self.modality_temp,       # NEW: Cross-modal fusion
        )

        # ========== Vector Encoders (with FiLM) ==========
        # Use config.max_time_steps for vector encoder positional encoding
        # Both image and vector modalities now use the same temporal PE max_len

        self.evap_encoder = VectorModalityEncoder(
            in_feat=1,
            stat_dim=config.static_attr_dim,
            d_model=config.d_model,
            n_layers=config.vec_encoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            max_len=config.max_time_steps,  # Same as image encoder
            use_weighted_fm=config.use_weighted_fm,  # Phase 2
            use_fm_layers=config.use_fm_layers,      # Phase 2
            use_input=config.use_input,              # Phase 2
            patch_size=config.vector_patch_size,     # NEW: Vector patch size
            modality_token=self.modality_evap,       # NEW: Cross-modal fusion
        )

        self.riverflow_encoder = VectorModalityEncoder(
            in_feat=1,
            stat_dim=config.static_attr_dim,
            d_model=config.d_model,
            n_layers=config.vec_encoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            max_len=config.max_time_steps,  # Same as image encoder
            use_weighted_fm=config.use_weighted_fm,  # Phase 2
            use_fm_layers=config.use_fm_layers,      # Phase 2
            use_input=config.use_input,              # Phase 2
            patch_size=config.vector_patch_size,     # NEW: Vector patch size
            modality_token=self.modality_riverflow,  # NEW: Cross-modal fusion
        )

        # ========== Shared Fusion Transformer ==========
        # 参考CAV-MAE的blocks_u (unified branch)
        # 让多个模态的visible tokens互相交互
        self.shared_depth = getattr(config, 'shared_depth', 1)  # 默认1层

        self.blocks_shared = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=4 * config.d_model,
                dropout=config.dropout,
                batch_first=True,
            )
            for _ in range(self.shared_depth)
        ])

        # Normalization for fused features
        self.norm_shared = nn.LayerNorm(config.d_model)

        # ========== Image Decoders ==========
        num_patches = (config.image_height // config.patch_size) * \
                     (config.image_width // config.patch_size)

        self.precip_decoder = ImageModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            num_patches=num_patches,
            patch_dim=config.patch_size * config.patch_size,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            num_encoder_layers=config.img_encoder_layers,  # NEW: Phase 2
        )

        self.soil_decoder = ImageModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            num_patches=num_patches,
            patch_dim=config.patch_size * config.patch_size,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            num_encoder_layers=config.img_encoder_layers,  # NEW: Phase 2
        )

        self.temp_decoder = ImageModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            num_patches=num_patches,
            patch_dim=config.patch_size * config.patch_size,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            max_time_steps=config.max_time_steps,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # NEW: Phase 2
            num_encoder_layers=config.img_encoder_layers,  # NEW: Phase 2
        )

        # ========== Vector Decoders ==========
        self.evap_decoder = VectorModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            max_time_steps=config.max_time_steps,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # Phase 2
            num_encoder_layers=config.vec_encoder_layers,  # Phase 2
        )

        self.riverflow_decoder = VectorModalityDecoder(
            encoder_dim=config.d_model,
            decoder_dim=config.decoder_dim,
            max_time_steps=config.max_time_steps,
            num_decoder_layers=config.decoder_layers,
            nhead=config.nhead,
            dropout=config.dropout,
            use_cross_attn=config.use_cross_attn,  # CrossMAE config
            decoder_self_attn=config.decoder_self_attn,
            use_weighted_fm=config.use_weighted_fm,  # Phase 2
            num_encoder_layers=config.vec_encoder_layers,  # Phase 2
        )

        # ========== Initialize modality tokens ==========
        # Encoder modality tokens
        nn.init.normal_(self.modality_precip, std=0.02)
        nn.init.normal_(self.modality_soil, std=0.02)
        nn.init.normal_(self.modality_temp, std=0.02)
        nn.init.normal_(self.modality_evap, std=0.02)
        nn.init.normal_(self.modality_riverflow, std=0.02)

        # Decoder modality tokens
        nn.init.normal_(self.decoder_modality_precip, std=0.02)
        nn.init.normal_(self.decoder_modality_soil, std=0.02)
        nn.init.normal_(self.decoder_modality_temp, std=0.02)
        nn.init.normal_(self.decoder_modality_evap, std=0.02)
        nn.init.normal_(self.decoder_modality_riverflow, std=0.02)

    def forward(self, batch: Dict) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass

        Args:
            batch: Dictionary with:
                - 'precip', 'soil', 'temp': [B, T, H, W] image data
                - 'evap', 'riverflow': [B, T] vector data
                - 'static_attr': [B, stat_dim] static attributes
                - '*_mask': [B, T, num_patches] or [B, T] masks (True=masked)

        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary of individual losses for each modality
        """

        # ===== Encode all modalities =====
        # Image modalities
        precip_token, precip_mask_info = self.precip_encoder(
            batch['precip'], batch['precip_mask']
        )
        soil_token, soil_mask_info = self.soil_encoder(
            batch['soil'], batch['soil_mask']
        )
        temp_token, temp_mask_info = self.temp_encoder(
            batch['temp'], batch['temp_mask']
        )

        # Vector modalities (with FiLM)
        evap_token, evap_mask_info = self.evap_encoder(
            batch['evap'], batch['static_attr'], batch['evap_mask']
        )

        # ⭐ NEW: Extract riverflow_valid_mask and pass to riverflow encoder
        riverflow_valid_mask = batch.get('riverflow_valid_mask', None)
        if riverflow_valid_mask is not None:
            riverflow_valid_mask = riverflow_valid_mask.to(batch['precip'].device)

        riverflow_token, riverflow_mask_info = self.riverflow_encoder(
            batch['riverflow'],
            batch['static_attr'],
            batch['riverflow_mask'],
            valid_mask=riverflow_valid_mask  # ⭐ NEW: Pass valid_mask to encoder
        )

        # ===== Shared Fusion Layers =====
        # Step 1: 获取batch size和device
        B = precip_token.shape[0]
        device = precip_token.device

        # Step 2: 拼接所有模态的visible tokens
        # 保留 Vector modality 的 static token（包含重要的全局静态环境信息）
        all_tokens = torch.cat([
            precip_token,              # [B, L_precip, d_model]
            soil_token,                # [B, L_soil, d_model]
            temp_token,                # [B, L_temp, d_model]
            evap_token,                # [B, L_evap, d_model] 保留 static token
            riverflow_token            # [B, L_river, d_model] 保留 static token
        ], dim=1)  # [B, L_total, d_model]

        # Step 3: 创建padding mask (拼接各自的padding mask)
        # 从mask_info中获取padding_mask，如果没有则创建全False的mask
        precip_pad = precip_mask_info.get('padding_mask',
            torch.zeros(B, precip_token.shape[1], device=device, dtype=torch.bool))
        soil_pad = soil_mask_info.get('padding_mask',
            torch.zeros(B, soil_token.shape[1], device=device, dtype=torch.bool))
        temp_pad = temp_mask_info.get('padding_mask',
            torch.zeros(B, temp_token.shape[1], device=device, dtype=torch.bool))
        evap_pad = evap_mask_info.get('padding_mask',
            torch.zeros(B, evap_token.shape[1], device=device, dtype=torch.bool))
        riverflow_pad = riverflow_mask_info.get('padding_mask',
            torch.zeros(B, riverflow_token.shape[1], device=device, dtype=torch.bool))

        # 保留完整的 padding mask（包括 static token 的 mask）
        all_padding_mask = torch.cat([
            precip_pad,
            soil_pad,
            temp_pad,
            evap_pad,          # 包含 static token 的 mask
            riverflow_pad      # 包含 static token 的 mask
        ], dim=1)  # [B, L_total]

        # Step 4: 通过shared transformer进行跨模态融合
        fused_features = all_tokens
        for blk in self.blocks_shared:
            fused_features = blk(fused_features, src_key_padding_mask=all_padding_mask)
        fused_features = self.norm_shared(fused_features)
        # fused_features: [B, L_total, d_model] - 融合后的multi-modal features

        # ===== Decode all modalities =====
        # ⭐ 所有decoder现在接收fused_features（而非单模态token）
        # ⭐ NEW: Pass global padding_mask to all decoders for cross-attention

        # Image modalities
        precip_pred = self.precip_decoder(
            fused_features,
            precip_mask_info,
            decoder_modality_token=self.decoder_modality_precip,
            encoder_padding_mask=all_padding_mask  # ⭐ NEW: Global padding mask
        )
        soil_pred = self.soil_decoder(
            fused_features,
            soil_mask_info,
            decoder_modality_token=self.decoder_modality_soil,
            encoder_padding_mask=all_padding_mask  # ⭐ NEW: Global padding mask
        )
        temp_pred = self.temp_decoder(
            fused_features,
            temp_mask_info,
            decoder_modality_token=self.decoder_modality_temp,
            encoder_padding_mask=all_padding_mask  # ⭐ NEW: Global padding mask
        )

        # Vector modalities
        evap_pred = self.evap_decoder(
            fused_features,
            evap_mask_info,
            decoder_modality_token=self.decoder_modality_evap,
            encoder_padding_mask=all_padding_mask  # ⭐ NEW: Global padding mask
        )
        riverflow_pred = self.riverflow_decoder(
            fused_features,
            riverflow_mask_info,
            decoder_modality_token=self.decoder_modality_riverflow,
            encoder_padding_mask=all_padding_mask  # ⭐ NEW: Global padding mask
        )

        # ===== Compute losses =====
        loss_dict = {}

        # Image losses
        loss_dict['precip_loss'] = self._compute_image_loss(
            precip_pred, batch['precip'], batch['precip_mask']
        )
        loss_dict['soil_loss'] = self._compute_image_loss(
            soil_pred, batch['soil'], batch['soil_mask']
        )
        loss_dict['temp_loss'] = self._compute_image_loss(
            temp_pred, batch['temp'], batch['temp_mask']
        )

        # Extract riverflow valid mask [B] (bool: True=valid, False=invalid)
        riverflow_valid_mask = batch.get('riverflow_valid_mask', None)  # [B]
        device = batch['precip'].device
        if riverflow_valid_mask is not None:
            riverflow_valid_mask = riverflow_valid_mask.to(device)
        else:
            # If not provided (e.g., test scripts), assume all valid
            riverflow_valid_mask = torch.ones(batch['precip'].shape[0], device=device, dtype=torch.bool)

        # Vector losses
        loss_dict['evap_loss'] = self._compute_vector_loss(
            evap_pred, batch['evap'], batch['evap_mask']
        )
        loss_dict['riverflow_loss'] = self._compute_vector_loss(
            riverflow_pred, batch['riverflow'], batch['riverflow_mask'],
            valid_sample_mask=riverflow_valid_mask  # Pass per-sample validity
        )

        # Total loss with task weights
        total_loss = 0.0
        task_weights = getattr(self.config, 'task_weights', {})

        for key, value in loss_dict.items():
            weight = task_weights.get(key, 1.0)  # Default weight is 1.0
            total_loss += weight * value

        loss_dict['total_loss'] = total_loss

        return total_loss, loss_dict

    def _compute_image_loss(
        self,
        pred_patches: Tensor,
        target_img: Tensor,
        mask: Tensor
    ) -> Tensor:
        """
        Compute reconstruction loss for image modality

        Args:
            pred_patches: [B, T, num_patches, patch_dim] predicted patches
            target_img: [B, T, H, W] target image
            mask: [B, T, num_patches] bool mask (True=masked)

        Returns:
            Scalar loss (only on masked AND valid land patches)
        """
        # Patchify target
        target_patches = patchify(
            target_img,
            patch_size=self.config.patch_size
        )  # [B, T, num_patches, patch_dim]

        # MSE loss
        loss = F.mse_loss(
            pred_patches, target_patches, reduction='none'
        )  # [B, T, num_patches, patch_dim]

        # Average over patch dimension
        loss = loss.mean(dim=-1)  # [B, T, num_patches]

        # Create valid patch mask [1, 1, num_patches]
        # Only 94 patches are valid land patches
        valid_mask = torch.zeros(1, 1, pred_patches.shape[2], device=loss.device)
        valid_mask[0, 0, self.valid_patch_indices] = 1.0

        # Combine with temporal mask: only compute loss on masked AND valid patches
        # mask: [B, T, num_patches] - True for masked positions
        # valid_mask: [1, 1, num_patches] - 1.0 for valid land patches
        combined_mask = mask.float() * valid_mask  # [B, T, num_patches]

        # Only compute loss on masked AND valid patches
        # Use 1e-6 instead of 1e-8 for FP16 compatibility (1e-8 rounds to 0 in FP16)
        masked_loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-6)

        return masked_loss

    def _compute_vector_loss(
        self,
        pred_vec: Tensor,
        target_vec: Tensor,
        mask: Tensor,
        valid_sample_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute reconstruction loss for vector modality

        Args:
            pred_vec: [B, num_catchments, T] predicted values (unpatchified)
            target_vec: [B, num_patches, patch_size, T] target values (patchified)
            mask: [B, num_patches, T] bool mask (True=masked patch)
            valid_sample_mask: [B] bool mask (True=valid sample, False=invalid/missing data)

        Returns:
            Scalar loss (only on masked positions from valid samples)
        """
        # Unpatchify target: [B, num_patches, patch_size, T] -> [B, num_catchments, T]
        B, num_patches, patch_size, T = target_vec.shape
        num_padded = num_patches * patch_size

        # Flatten patch dimension: [B, num_patches, patch_size, T] -> [B, num_padded, T]
        target_flat = target_vec.reshape(B, num_padded, T)

        # Remove padding to match actual number of catchments
        num_actual = pred_vec.shape[1]  # Should be 604
        target_unpatch = target_flat[:, :num_actual, :]  # [B, 604, T]

        # Unpatchify mask: [B, num_patches, T] -> [B, num_catchments, T]
        # Expand mask to cover all catchments in each patch
        mask_expanded = mask.unsqueeze(2).expand(-1, -1, patch_size, -1)  # [B, num_patches, patch_size, T]
        mask_flat = mask_expanded.reshape(B, num_padded, T)  # [B, num_padded, T]
        mask_unpatch = mask_flat[:, :num_actual, :]  # [B, 604, T]

        # MSE loss
        loss = F.mse_loss(
            pred_vec, target_unpatch, reduction='none'
        )  # [B, num_catchments, T]

        # Apply valid_sample_mask if provided
        if valid_sample_mask is not None:
            # Expand to match loss shape: [B] -> [B, num_catchments, T]
            valid_mask_expanded = valid_sample_mask.view(B, 1, 1).expand(-1, num_actual, T)  # [B, 604, T]
            # Combine with mask: only compute loss on masked AND valid positions
            combined_mask = mask_unpatch.float() * valid_mask_expanded.float()
        else:
            combined_mask = mask_unpatch.float()

        # Only compute loss on masked positions from valid samples
        # Use 1e-6 instead of 1e-8 for FP16 compatibility
        masked_loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-6)

        return masked_loss


if __name__ == '__main__':
    """Unit test for MultiModalMAE"""

    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from models.image_encoder import ImageModalityEncoder
    from models.vector_encoder import VectorModalityEncoder
    from models.image_decoder import ImageModalityDecoder
    from models.vector_decoder import VectorModalityDecoder
    from models.layers import patchify
    from configs.mae_config import MAEConfig

    print("=" * 60)
    print("Testing MultiModalMAE")
    print("=" * 60)

    # Create config
    config = MAEConfig()

    # Simulate valid patch indices
    num_valid = 94
    valid_patch_indices = torch.randperm(522)[:num_valid].sort()[0]

    # Create model
    model = MultiModalMAE(config, valid_patch_indices)

    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create test batch (smaller size to avoid OOM)
    B, T, H, W = 2, 10, 290, 180
    batch = {
        # Image modalities
        'precip': torch.randn(B, T, H, W),
        'soil': torch.randn(B, T, H, W),
        'temp': torch.randn(B, T, H, W),

        # Vector modalities
        'evap': torch.randn(B, T),
        'riverflow': torch.randn(B, T),

        # Static attributes
        'static_attr': torch.randn(B, 11),

        # Masks (75% masked)
        'precip_mask': torch.rand(B, T, 522) < 0.75,
        'soil_mask': torch.rand(B, T, 522) < 0.75,
        'temp_mask': torch.rand(B, T, 522) < 0.75,
        'evap_mask': torch.rand(B, T) < 0.75,
        'riverflow_mask': torch.rand(B, T) < 0.75,
    }

    print(f"\n✓ Test batch created")
    print(f"  Image shape: {batch['precip'].shape}")
    print(f"  Vector shape: {batch['evap'].shape}")

    # Forward pass
    total_loss, loss_dict = model(batch)

    print(f"\n✓ Forward pass successful")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Individual losses:")
    for key, value in loss_dict.items():
        if key != 'total_loss':
            print(f"    {key}: {value.item():.4f}")

    # Backward pass
    total_loss.backward()

    print(f"\n✓ Backward pass successful")

    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"✓ Parameters with gradients: {has_grad}/{total_params}")

    print(f"\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 60)
