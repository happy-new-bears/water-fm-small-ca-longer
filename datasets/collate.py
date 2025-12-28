"""
Collate function for MAE-style masked training
"""

import torch
import numpy as np
from typing import Dict, List


class MultiScaleMaskedCollate:
    """
    Collate function for MAE-style masked pretraining.

    Features:
    - Fixed sequence length
    - Patch-level masking for images (like ViT MAE)
    - Temporal masking for vectors
    - Support for unified/independent mask strategies across modalities
    """

    def __init__(
        self,
        # Sequence length (fixed)
        seq_len: int = 90,
        # Mask parameters
        mask_ratio: float = 0.75,  # Ratio to mask (0.75 like MAE paper)
        # Image patch parameters
        patch_size: int = 10,  # Each patch is 10x10 pixels for images
        image_height: int = 290,
        image_width: int = 180,
        # Vector patch parameters (spatial patchify)
        vector_patch_size: int = 8,  # Each spatial patch has 8 catchments
        # Land mask
        land_mask_path: str = None,  # Path to land mask file
        land_threshold: float = 0.5,  # Minimum land ratio for valid patch
        # Modality mask strategy
        mask_mode: str = 'unified',  # 'independent' or 'unified'
        # Mode
        mode: str = 'train',  # 'train', 'val', 'test'
    ):
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.vector_patch_size = vector_patch_size
        self.image_height = image_height
        self.image_width = image_width
        self.land_threshold = land_threshold
        self.mask_mode = mask_mode
        self.mode = mode

        # Calculate number of patches
        self.num_patches_h = image_height // patch_size  # 290 // 10 = 29
        self.num_patches_w = image_width // patch_size   # 180 // 10 = 18
        self.num_patches = self.num_patches_h * self.num_patches_w  # 29 * 18 = 522

        # Load and process land mask
        self.valid_patch_indices = self._process_land_mask(land_mask_path)

        # Modality lists
        self.image_modalities = ['precip', 'soil', 'temp']
        self.vector_modalities = ['evap', 'riverflow']
        self.all_modalities = self.image_modalities + self.vector_modalities

        # For validation: use a dedicated RNG with fixed seed for reproducible masking
        if self.mode in ['val', 'test']:
            self.val_rng = np.random.RandomState(42)  # Fixed seed for validation
        else:
            self.val_rng = None

    def _process_land_mask(self, land_mask_path: str) -> np.ndarray:
        """
        Process land mask to identify valid patches

        Args:
            land_mask_path: Path to land mask file (290x180)

        Returns:
            valid_patch_indices: Array of valid patch indices
        """
        if land_mask_path is None:
            # If no land mask provided, all patches are valid
            return np.arange(self.num_patches)

        import torch
        # Load land mask
        land_mask = torch.load(land_mask_path).numpy()  # [290, 180]

        # Calculate land coverage for each patch
        valid_patches = []
        patch_idx = 0

        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # Extract patch region
                patch = land_mask[
                    i*self.patch_size:(i+1)*self.patch_size,
                    j*self.patch_size:(j+1)*self.patch_size
                ]

                # Calculate land ratio in this patch
                land_ratio = patch.sum() / (self.patch_size * self.patch_size)

                # Check if patch meets threshold
                if land_ratio >= self.land_threshold:
                    valid_patches.append(patch_idx)

                patch_idx += 1

        valid_patches = np.array(valid_patches, dtype=np.int64)
        print(f"Land mask loaded: {len(valid_patches)}/{self.num_patches} patches are valid "
              f"(>={self.land_threshold*100:.0f}% land)")

        return valid_patches

    def __call__(self, batch_list: List[Dict]) -> Dict:
        """
        Process batch

        Args:
            batch_list: List of samples from Dataset

        Returns:
            batch_dict: {
                # Data
                'precip': [B, T, 290, 180],
                'soil': [B, T, 290, 180],
                'temp': [B, T, 290, 180],
                'evap': [B, T],
                'riverflow': [B, T],
                'static_attr': [B, num_features],

                # Masks (True = positions to predict)
                # For images: [B, T, num_patches] where num_patches = 522
                'precip_mask': [B, T, 522],
                'soil_mask': [B, T, 522],
                'temp_mask': [B, T, 522],
                # For vectors: [B, T] (temporal masking)
                'evap_mask': [B, T],
                'riverflow_mask': [B, T],

                # Metadata
                'catchment_ids': [B],
                'seq_len': int,
            }
        """
        B = len(batch_list)
        seq_len = self.seq_len

        # Step 1: Truncate/pad data to fixed sequence length and extract metadata
        truncated_batch = []
        num_vec_patches = None  # Will be set from first sample
        for sample in batch_list:
            truncated = {}
            for key, val in sample.items():
                if key in self.image_modalities:
                    # Image: truncate/pad time dimension [T, H, W]
                    T = val.shape[0]
                    if T >= seq_len:
                        truncated[key] = val[:seq_len]
                    else:
                        # Pad with zeros if too short
                        if isinstance(val, torch.Tensor):
                            pad_shape = (seq_len - T, *val.shape[1:])
                            padding = torch.zeros(pad_shape, dtype=val.dtype, device=val.device)
                            truncated[key] = torch.cat([val, padding], dim=0)
                        else:
                            pad_shape = (seq_len - T, *val.shape[1:])
                            padding = np.zeros(pad_shape, dtype=val.dtype)
                            truncated[key] = np.concatenate([val, padding], axis=0)
                elif key in self.vector_modalities:
                    # Vector patches: truncate/pad time dimension [num_patches, patch_size, T]
                    T = val.shape[2]
                    if T >= seq_len:
                        truncated[key] = val[:, :, :seq_len]
                    else:
                        # Pad with zeros if too short
                        if isinstance(val, torch.Tensor):
                            pad_shape = (*val.shape[:2], seq_len - T)
                            padding = torch.zeros(pad_shape, dtype=val.dtype, device=val.device)
                            truncated[key] = torch.cat([val, padding], dim=2)
                        else:
                            pad_shape = (*val.shape[:2], seq_len - T)
                            padding = np.zeros(pad_shape, dtype=val.dtype)
                            truncated[key] = np.concatenate([val, padding], axis=2)

                    if num_vec_patches is None:
                        num_vec_patches = val.shape[0]  # Get number of patches
                else:
                    truncated[key] = val
            truncated_batch.append(truncated)

        # Step 2: Generate masks
        # Both train and val use masking, but val uses fixed seed for reproducibility
        masks = self._generate_masks(B, seq_len, num_vec_patches)

        # Step 3: Stack into batch
        batch_dict = {}

        # Image modalities
        for mod in self.image_modalities:
            # Handle both numpy arrays and torch tensors
            batch_dict[mod] = torch.stack([
                s[mod] if isinstance(s[mod], torch.Tensor) else torch.from_numpy(s[mod])
                for s in truncated_batch
            ]).float()  # [B, T, 290, 180]
            batch_dict[f'{mod}_mask'] = torch.from_numpy(masks[mod])  # [B, T, num_patches]

        # Vector modalities (patch-level)
        for mod in self.vector_modalities:
            # Handle both numpy arrays and torch tensors
            batch_dict[mod] = torch.stack([
                s[mod] if isinstance(s[mod], torch.Tensor) else torch.from_numpy(s[mod])
                for s in truncated_batch
            ]).float()  # [B, num_patches, patch_size, T]
            batch_dict[f'{mod}_mask'] = torch.from_numpy(masks[mod])  # [B, num_patches, T]

        # Static attributes (patch-level)
        batch_dict['static_attr'] = torch.stack([
            s['static_attr'] for s in truncated_batch
        ])  # [B, num_patches, patch_size, num_features]

        # Catchment padding mask
        batch_dict['catchment_padding_mask'] = torch.stack([
            s['catchment_padding_mask'] for s in truncated_batch
        ])  # [B, num_patches, patch_size]

        # Metadata
        batch_dict['seq_len'] = seq_len
        batch_dict['num_vec_patches'] = num_vec_patches
        batch_dict['vector_patch_size'] = truncated_batch[0]['patch_size']

        # ⭐ NEW: Collect riverflow_valid flags from all samples
        riverflow_valid_list = []
        for sample in truncated_batch:
            # Default to True for backward compatibility if field doesn't exist
            riverflow_valid_list.append(sample.get('riverflow_valid', True))

        batch_dict['riverflow_valid_mask'] = torch.tensor(
            riverflow_valid_list,
            dtype=torch.bool
        )  # [B]

        return batch_dict

    def _generate_masks(self, B: int, seq_len: int, num_vec_patches: int) -> Dict[str, np.ndarray]:
        """
        Generate masks for each modality

        Args:
            B: batch size
            seq_len: sequence length
            num_vec_patches: number of spatial patches for vectors

        Returns:
            Dictionary with:
            - Image modalities: mask [B, T, num_patches]
            - Vector modalities: mask [B, num_patches, T]
        """
        masks = {}

        if self.mask_mode == 'unified':
            # All modalities use the same mask pattern
            # For images: generate patch-level mask [B, T, num_patches]
            image_mask = self._generate_image_mask(B, seq_len)
            for mod in self.image_modalities:
                masks[mod] = image_mask.copy()

            # For vectors: generate patch-level temporal mask [B, num_patches, T]
            vector_mask = self._generate_vector_mask(B, seq_len, num_vec_patches)
            for mod in self.vector_modalities:
                masks[mod] = vector_mask.copy()

        elif self.mask_mode == 'independent':
            # Each modality has independent mask
            for mod in self.image_modalities:
                masks[mod] = self._generate_image_mask(B, seq_len)
            for mod in self.vector_modalities:
                masks[mod] = self._generate_vector_mask(B, seq_len, num_vec_patches)

        return masks

    def _generate_image_mask(self, B: int, seq_len: int) -> np.ndarray:
        """
        Generate patch-level mask for images (MAE-style)
        Only masks valid patches (land patches)

        Args:
            B: batch size
            seq_len: sequence length

        Returns:
            mask: [B, T, num_patches], True = positions to predict
        """
        masks = []

        for _ in range(B):
            # For each sample, generate mask for each timestep
            sample_masks = []
            for _ in range(seq_len):
                # Initialize mask: False for all patches
                patch_mask = np.zeros(self.num_patches, dtype=bool)

                # Only mask from valid patches
                num_valid = len(self.valid_patch_indices)
                if num_valid > 0:
                    # Calculate how many valid patches to mask
                    num_to_mask = int(num_valid * self.mask_ratio)
                    num_to_mask = max(1, num_to_mask)  # At least mask 1 patch

                    # Randomly select which valid patches to mask
                    # Use val_rng for validation (fixed seed), otherwise use global np.random
                    if self.val_rng is not None:
                        masked_valid_indices = self.val_rng.choice(
                            num_valid,
                            size=num_to_mask,
                            replace=False
                        )
                    else:
                        masked_valid_indices = np.random.choice(
                            num_valid,
                            size=num_to_mask,
                            replace=False
                        )

                    # Convert to actual patch indices
                    masked_patch_indices = self.valid_patch_indices[masked_valid_indices]

                    # Set mask
                    patch_mask[masked_patch_indices] = True

                sample_masks.append(patch_mask)

            masks.append(np.stack(sample_masks, axis=0))  # [T, num_patches]

        return np.stack(masks, axis=0)  # [B, T, num_patches]

    def _generate_vector_mask(self, B: int, seq_len: int, num_patches: int) -> np.ndarray:
        """
        Generate patch-level temporal mask for vectors (similar to image masking)

        Randomly selects mask_ratio of (patch, time) combinations to mask.

        Args:
            B: batch size
            seq_len: sequence length
            num_patches: number of spatial patches

        Returns:
            mask: [B, num_patches, T], True = patch-time positions to predict
        """
        masks = []

        for _ in range(B):
            # Generate patch-level temporal mask: [num_patches, T]
            # Similar to image masking strategy but for spatial patches over time

            # Create mask for each time step
            sample_masks = []
            for _ in range(seq_len):
                # Randomly select which patches to mask at this timestep
                num_to_mask = int(num_patches * self.mask_ratio)
                patch_mask = np.zeros(num_patches, dtype=bool)

                # Use val_rng for validation (fixed seed), otherwise use global np.random
                if self.val_rng is not None:
                    masked_patch_indices = self.val_rng.choice(
                        num_patches,
                        size=num_to_mask,
                        replace=False
                    )
                else:
                    masked_patch_indices = np.random.choice(
                        num_patches,
                        size=num_to_mask,
                        replace=False
                    )

                patch_mask[masked_patch_indices] = True
                sample_masks.append(patch_mask)

            # Stack to [T, num_patches], then transpose to [num_patches, T]
            mask_2d = np.stack(sample_masks, axis=0).T  # [num_patches, T]
            masks.append(mask_2d)

        return np.stack(masks, axis=0)  # [B, num_patches, T]
