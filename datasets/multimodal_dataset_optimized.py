"""
Ultra-optimized Multi-modal Hydrology Dataset with pre-normalization

Performance optimizations:
1. Pre-normalize all vector data in __init__ (done once)
2. Pre-normalize all static attributes in __init__
3. Cache static stats to avoid recomputation
4. Direct numpy slicing in __getitem__ (no loops)
5. Vectorized patchify using reshape (no loops)

Speed: 10-100x faster than original implementation
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset


class MultiModalHydroDatasetOptimized(Dataset):
    """
    Ultra-optimized Multi-modal Hydrology Dataset

    Key features:
    - Pre-normalized data (compute once, use many times)
    - Vectorized patchify (no loops)
    - Cached statistics (including static attrs)
    - Direct slicing in __getitem__
    """

    def __init__(
        self,
        # Merged h5 files (optimized mode)
        precip_h5: str,
        soil_h5: str,
        temp_h5: str,
        # Vector modality data (pre-loaded)
        evap_data: np.ndarray,  # [num_catchments, num_days]
        riverflow_data: np.ndarray,  # [num_catchments, num_days]
        # Static attributes
        static_attr_file: str,
        static_attr_vars: List[str],
        # Time range
        start_date: str,
        end_date: str,
        # Sampling parameters
        max_sequence_length: int = 90,
        stride: int = 30,
        # Catchment configuration
        catchment_ids: Optional[np.ndarray] = None,
        # Normalization
        stats_cache_path: Optional[str] = None,
        land_mask_path: Optional[str] = None,
        # Patchify parameters
        patch_size: int = 8,
        # Other
        split: str = 'train',
        cache_to_memory: bool = True,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.stride = stride
        self.split = split
        self.static_attr_vars = static_attr_vars
        self.cache_to_memory = cache_to_memory
        self.patch_size = patch_size

        print(f"\n{'='*60}")
        print(f"Initializing ULTRA-OPTIMIZED dataset (split: {split})")
        print(f"{'='*60}")

        # Generate date list
        self.date_list = self._generate_date_list(start_date, end_date)
        self.num_days = len(self.date_list)
        print(f"Date range: {self.date_list[0]} to {self.date_list[-1]} ({self.num_days} days)")

        # Store catchment IDs
        self.num_catchments = evap_data.shape[0]
        if catchment_ids is not None:
            if len(catchment_ids) != self.num_catchments:
                raise ValueError(
                    f"Length of catchment_ids ({len(catchment_ids)}) must match "
                    f"first dimension of data ({self.num_catchments})"
                )
            self.catchment_ids = catchment_ids
        else:
            self.catchment_ids = np.arange(self.num_catchments)

        # Calculate patchify dimensions
        self.num_patches = (self.num_catchments + patch_size - 1) // patch_size
        self.num_padded = self.num_patches * patch_size
        print(f"Patchify: {self.num_catchments} catchments -> {self.num_patches} patches (size={patch_size})")

        # Load merged h5 files
        self.image_data = self._load_merged_h5_files(
            precip_h5, soil_h5, temp_h5
        )

        # Load static attributes
        self.static_attrs = self._load_static_attributes(
            static_attr_file, self.catchment_ids, static_attr_vars
        )

        # Load or compute normalization stats (INCLUDING STATIC STATS!)
        if stats_cache_path and os.path.exists(stats_cache_path):
            print(f"\n{'='*60}")
            print(f"Loading cached normalization stats from:")
            print(f"  {stats_cache_path}")
            print(f"{'='*60}")
            self.stats = torch.load(stats_cache_path)
            print("✓ Stats loaded successfully")
        else:
            print(f"\n{'='*60}")
            print("Computing normalization stats (this may take a while)...")
            print(f"{'='*60}")
            if land_mask_path is None:
                raise ValueError("land_mask_path required when computing stats")
            self.stats = self._compute_all_stats(land_mask_path, evap_data, riverflow_data)

            if stats_cache_path:
                print(f"\nSaving stats to: {stats_cache_path}")
                os.makedirs(os.path.dirname(stats_cache_path), exist_ok=True)
                torch.save(self.stats, stats_cache_path)
                print("✓ Stats saved successfully")

        # ===== KEY OPTIMIZATION: Pre-normalize ALL data =====
        print(f"\n{'='*60}")
        print("Pre-normalizing all data (done once, used many times)...")
        print(f"{'='*60}")

        # Pre-normalize vector data
        print("  Normalizing evaporation data...")
        self.evap_data_norm = self._prenormalize_vector_data(
            evap_data, 'evap'
        )  # [num_catchments, num_days]

        print("  Normalizing riverflow data...")
        self.riverflow_data_norm = self._prenormalize_vector_data(
            riverflow_data, 'riverflow'
        )  # [num_catchments, num_days]

        # Pre-normalize static attributes
        print("  Normalizing static attributes...")
        self.static_attrs_norm = self._prenormalize_static_attrs()  # [num_catchments, stat_dim]

        print("✓ All data pre-normalized!")

        # Build valid sample indices
        print(f"\n{'='*60}")
        print("Building valid sample indices...")
        print(f"{'='*60}")
        self.valid_samples = self._build_valid_samples(riverflow_data)
        print(f"✓ Found {len(self.valid_samples)} valid samples")

        print(f"\n{'='*60}")
        print(f"Dataset initialization complete!")
        print(f"  Catchments: {self.num_catchments}")
        print(f"  Days: {self.num_days}")
        print(f"  Valid samples: {len(self.valid_samples)}")
        print(f"  Patches: {self.num_patches}")
        print(f"{'='*60}\n")

    def _generate_date_list(self, start_date: str, end_date: str) -> List[datetime]:
        """Generate continuous date list"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        date_list = []
        current = start
        while current <= end:
            date_list.append(current)
            current += timedelta(days=1)

        return date_list

    def _load_merged_h5_files(
        self,
        precip_h5: str,
        soil_h5: str,
        temp_h5: str
    ) -> Dict[str, np.ndarray]:
        """Load merged h5 files"""
        image_data = {}

        for modality, h5_path in [
            ('precip', precip_h5),
            ('soil', soil_h5),
            ('temp', temp_h5)
        ]:
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"H5 file not found: {h5_path}")

            print(f"\nLoading {modality} from: {h5_path}")

            with h5py.File(h5_path, 'r') as f:
                if self.cache_to_memory:
                    data = f['data'][:]
                    print(f"  ✓ Cached to memory: {data.shape}, {data.nbytes / (1024**2):.2f} MB")
                else:
                    # Keep file handle open
                    if not hasattr(self, '_h5_handles'):
                        self._h5_handles = {}
                    self._h5_handles[modality] = h5py.File(h5_path, 'r')
                    data = self._h5_handles[modality]['data']
                    print(f"  ✓ File opened (on-demand loading)")

                image_data[modality] = data

        return image_data

    def _load_static_attributes(
        self,
        file_path: str,
        catchment_ids: np.ndarray,
        vars_to_use: List[str]
    ) -> torch.Tensor:
        """Load static attributes from CSV"""
        df = pd.read_csv(file_path)
        id_col = 'id'

        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' not found in {file_path}")

        df = df.drop_duplicates(subset=[id_col], keep='first')
        df_filtered = df[df[id_col].isin(catchment_ids)]

        missing = set(catchment_ids) - set(df_filtered[id_col])
        if missing:
            print(f"Warning: {len(missing)} catchments missing in attributes")

        df_ordered = df_filtered.set_index(id_col).loc[catchment_ids].reset_index()

        try:
            attrs = df_ordered[vars_to_use].values.astype(np.float32)
        except KeyError as e:
            raise KeyError(f"Attributes {e} not found. Available: {list(df.columns)}")

        if np.isnan(attrs).any():
            print("Warning: NaN in static attributes, filling with column means")
            col_means = np.nanmean(attrs, axis=0)
            for i in range(attrs.shape[1]):
                attrs[np.isnan(attrs[:, i]), i] = col_means[i]

        return torch.from_numpy(attrs)

    def _compute_all_stats(
        self,
        land_mask_path: str,
        evap_data: np.ndarray,
        riverflow_data: np.ndarray
    ) -> Dict:
        """
        Compute ALL normalization statistics

        IMPORTANT: Also computes and caches static attribute stats!
        """
        land_mask = torch.load(land_mask_path)
        stats = {'land_mask': land_mask}

        # 1. Image modality stats
        print("  Computing image statistics (land pixels only)...")
        for modality in ['precip', 'soil', 'temp']:
            img_stats = self._compute_image_stats(modality, land_mask, num_samples=1000)
            stats[f'{modality}_mean'] = img_stats['mean']
            stats[f'{modality}_std'] = img_stats['std']
            print(f"    {modality}: mean={img_stats['mean'].item():.4f}, std={img_stats['std'].item():.4f}")

        # 2. Vector modality stats (per-catchment)
        print("  Computing vector statistics (per-catchment)...")
        vec_stats = self._compute_vector_stats(evap_data, riverflow_data)
        stats.update(vec_stats)
        print(f"    evap: computed for {self.num_catchments} catchments")
        print(f"    riverflow: computed for {self.num_catchments} catchments")

        # 3. Static attribute stats (IMPORTANT: Cache these!)
        print("  Computing static attribute statistics (global)...")
        static_stats = self._compute_static_stats()
        stats['static_mean'] = static_stats['mean']
        stats['static_std'] = static_stats['std']
        print(f"    static: mean shape={static_stats['mean'].shape}, std shape={static_stats['std'].shape}")

        return stats

    def _compute_image_stats(
        self,
        modality: str,
        land_mask: torch.Tensor,
        num_samples: int = None  # None = use all timesteps
    ) -> Dict[str, torch.Tensor]:
        """
        Compute statistics for image modality (ONLY valid land values)

        Note: Changed to use all timesteps for accurate statistics
        """
        all_land_values = []

        # Use ALL timesteps for accurate statistics (not sampling)
        print(f"      Computing stats from all {self.num_days} timesteps...")

        for idx in range(self.num_days):
            img = self.image_data[modality][idx]

            # ✅ Get land values (corrected mask excludes missing pixels)
            land_values = img[land_mask.numpy() == 1]

            # ✅ Safety check - filter any remaining invalid values (should be none now)
            valid_mask = (land_values > -1000) & (land_values < 1e6) & (~np.isnan(land_values))
            valid_values = land_values[valid_mask]

            if len(valid_values) > 0:
                all_land_values.append(valid_values)

            # Progress indicator for large datasets
            if (idx + 1) % 1000 == 0:
                print(f"        Processed {idx + 1}/{self.num_days} timesteps...")

        if len(all_land_values) == 0:
            raise ValueError(f"No valid land values found for {modality}!")

        all_land_values = np.concatenate(all_land_values)
        mean = torch.tensor(all_land_values.mean(), dtype=torch.float32)
        std = torch.tensor(all_land_values.std(), dtype=torch.float32)

        print(f"      {modality}: computed from {len(all_land_values)} valid land pixels")
        print(f"        Mean: {mean.item():.4f}, Std: {std.item():.4f}")
        print(f"        Range: [{all_land_values.min():.2f}, {all_land_values.max():.2f}]")

        return {'mean': mean, 'std': std}

    def _compute_vector_stats(
        self,
        evap_data: np.ndarray,
        riverflow_data: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GLOBAL statistics for vector modalities

        Key changes:
        1. Evaporation: Global mean/std across all catchments (preserves relative magnitudes)
        2. Riverflow: Apply log transform first, then global mean/std (stabilizes extreme values)

        Args:
            evap_data: [num_catchments, num_days] evaporation data
            riverflow_data: [num_catchments, num_days] riverflow data

        Returns:
            Dictionary with global statistics (scalars, not per-catchment vectors)
        """
        # Evaporation: Global normalization across all catchments
        valid_evap = evap_data[~np.isnan(evap_data)]
        if len(valid_evap) > 0:
            evap_mean = float(valid_evap.mean())
            evap_std = float(valid_evap.std())
        else:
            evap_mean = 0.0
            evap_std = 1.0

        # Riverflow: Log transform + Global normalization
        # Apply log(x + eps) to handle zeros and stabilize large values (e.g., floods)
        eps = 1e-6
        riverflow_log = np.log(riverflow_data + eps)

        # Filter out invalid values (NaN and Inf)
        valid_river_log = riverflow_log[~np.isnan(riverflow_log) & ~np.isinf(riverflow_log)]
        if len(valid_river_log) > 0:
            riverflow_log_mean = float(valid_river_log.mean())
            riverflow_log_std = float(valid_river_log.std())
        else:
            riverflow_log_mean = 0.0
            riverflow_log_std = 1.0

        return {
            'evap_mean': torch.tensor(evap_mean, dtype=torch.float32),              # Scalar
            'evap_std': torch.tensor(evap_std, dtype=torch.float32),                # Scalar
            'riverflow_log_mean': torch.tensor(riverflow_log_mean, dtype=torch.float32),  # Scalar
            'riverflow_log_std': torch.tensor(riverflow_log_std, dtype=torch.float32),    # Scalar
        }

    def _compute_static_stats(self) -> Dict[str, torch.Tensor]:
        """
        Compute global statistics for static attributes

        IMPORTANT: This will be cached so we don't recompute every time!
        """
        return {
            'mean': self.static_attrs.mean(dim=0),  # [stat_dim]
            'std': self.static_attrs.std(dim=0),    # [stat_dim]
        }

    def _prenormalize_vector_data(
        self,
        data: np.ndarray,
        modality: str
    ) -> torch.Tensor:
        """
        Pre-normalize vector data using GLOBAL statistics

        Args:
            data: [num_catchments, num_days] numpy array
            modality: 'evap' or 'riverflow'

        Returns:
            normalized_data: [num_catchments, num_days] torch.Tensor
        """
        # Convert to torch first
        data_tensor = torch.from_numpy(data).float()

        if modality == 'evap':
            # Evaporation: Direct global normalization
            mean = self.stats['evap_mean']  # Scalar
            std = self.stats['evap_std']    # Scalar
            normalized = (data_tensor - mean) / (std + 1e-8)

        elif modality == 'riverflow':
            # Riverflow: Log transform + global normalization
            eps = 1e-6
            data_log = torch.log(data_tensor + eps)
            mean = self.stats['riverflow_log_mean']  # Scalar
            std = self.stats['riverflow_log_std']    # Scalar
            normalized = (data_log - mean) / (std + 1e-8)

        else:
            raise ValueError(f"Unknown modality: {modality}")

        return normalized

    def _prenormalize_static_attrs(self) -> torch.Tensor:
        """
        Pre-normalize static attributes (done once in __init__)

        Returns:
            normalized_static: [num_catchments, stat_dim]
        """
        mean = self.stats['static_mean']  # [stat_dim]
        std = self.stats['static_std']    # [stat_dim]

        normalized = (self.static_attrs - mean) / (std + 1e-8)

        return normalized

    def _build_valid_samples(
        self,
        riverflow_data: np.ndarray
    ) -> List[int]:
        """
        Build valid sample indices (time windows only)

        Each sample is a time window containing ALL catchments.
        No per-catchment duplication.

        Returns:
            List of day indices for valid time windows
        """
        valid_samples = []

        for day_idx in range(0, self.num_days - self.max_sequence_length + 1, self.stride):
            valid_samples.append(day_idx)

        return valid_samples

    def _normalize_image(
        self,
        img_seq: np.ndarray,
        modality: str
    ) -> torch.Tensor:
        """
        Normalize image sequence (land pixels only) - TORCH VERSION

        Args:
            img_seq: [T, H, W] numpy array
            modality: str (precip/soil/temp)

        Returns:
            img_norm: [T, H, W] torch.Tensor
        """
        mean = self.stats[f'{modality}_mean'].item()
        std = self.stats[f'{modality}_std'].item()
        land_mask = self.stats['land_mask']  # Already torch.Tensor [H, W]

        # ✅ Convert to torch
        img_tensor = torch.from_numpy(img_seq).float()  # [T, H, W]

        # ✅ Broadcast land_mask from [H, W] to [T, H, W]
        mask_3d = (land_mask == 1).unsqueeze(0)  # [1, H, W]

        # ✅ Normalize (corrected mask excludes missing pixels, so all land values are valid)
        img_norm = torch.where(
            mask_3d,
            (img_tensor - mean) / (std + 1e-8),
            torch.tensor(0.0)  # Ocean (and removed missing pixels) set to 0
        )

        return img_norm

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __del__(self):
        """Clean up h5 file handles"""
        if hasattr(self, '_h5_handles'):
            for handle in self._h5_handles.values():
                try:
                    handle.close()
                except:
                    pass

    def __getitem__(self, idx: int) -> Dict:
        """
        Ultra-fast __getitem__ using pre-normalized data and vectorized patchify

        Returns:
            {
                'precip': [T, 290, 180],
                'soil': [T, 290, 180],
                'temp': [T, 290, 180],
                'evap': [num_patches, patch_size, T],
                'riverflow': [num_patches, patch_size, T],
                'static_attr': [num_patches, patch_size, stat_dim],
                'catchment_padding_mask': [num_patches, patch_size],
                'num_patches': int,
                'patch_size': int,
                'start_date': datetime,
            }
        """
        start_day_idx = self.valid_samples[idx]
        end_day_idx = start_day_idx + self.max_sequence_length

        # 1. Image data: Direct slicing (super fast!)
        precip_seq = self.image_data['precip'][start_day_idx:end_day_idx]
        soil_seq = self.image_data['soil'][start_day_idx:end_day_idx]
        temp_seq = self.image_data['temp'][start_day_idx:end_day_idx]

        # Normalize images
        precip_norm = self._normalize_image(precip_seq, 'precip')
        soil_norm = self._normalize_image(soil_seq, 'soil')
        temp_norm = self._normalize_image(temp_seq, 'temp')

        # 2. Vector data: Direct slicing of PRE-NORMALIZED data (ultra fast!)
        evap_norm_all = self.evap_data_norm[:, start_day_idx:end_day_idx]  # [num_catchments, T] torch.Tensor
        riverflow_norm_all = self.riverflow_data_norm[:, start_day_idx:end_day_idx]  # [num_catchments, T] torch.Tensor

        # 3. Patchify using vectorized TORCH reshape (no loops!)
        # Pad if necessary
        if self.num_catchments % self.patch_size != 0:
            pad_size = self.num_padded - self.num_catchments
            T = evap_norm_all.shape[1]

            pad_zeros_evap = torch.zeros(pad_size, T, dtype=torch.float32)
            pad_zeros_river = torch.zeros(pad_size, T, dtype=torch.float32)

            evap_norm_all = torch.cat([evap_norm_all, pad_zeros_evap], dim=0)
            riverflow_norm_all = torch.cat([riverflow_norm_all, pad_zeros_river], dim=0)

        # Reshape: [num_padded, T] -> [num_patches, patch_size, T]
        evap_patches = evap_norm_all.reshape(self.num_patches, self.patch_size, -1)
        riverflow_patches = riverflow_norm_all.reshape(self.num_patches, self.patch_size, -1)

        # 4. Static attributes: Use PRE-NORMALIZED data
        static_norm_all = self.static_attrs_norm  # [num_catchments, stat_dim]

        # Pad static
        if self.num_catchments % self.patch_size != 0:
            pad_size = self.num_padded - self.num_catchments
            stat_dim = static_norm_all.shape[1]
            static_pad = torch.zeros(pad_size, stat_dim, dtype=torch.float32)
            static_norm_all = torch.cat([static_norm_all, static_pad], dim=0)

        # Reshape: [num_padded, stat_dim] -> [num_patches, patch_size, stat_dim]
        static_patches = static_norm_all.reshape(self.num_patches, self.patch_size, -1)

        # 5. Padding mask
        padding_mask = torch.zeros(self.num_patches, self.patch_size, dtype=torch.bool)
        if self.num_catchments % self.patch_size != 0:
            last_patch_valid = self.num_catchments % self.patch_size
            padding_mask[-1, last_patch_valid:] = True

        # 6. ⭐ NEW: Determine riverflow validity
        # Riverflow is only valid if the window starts on or after 1989-01-01
        start_date = self.date_list[start_day_idx]
        riverflow_valid = (start_date >= datetime(1989, 1, 1))

        return {
            'precip': precip_norm,
            'soil': soil_norm,
            'temp': temp_norm,
            'evap': evap_patches,
            'riverflow': riverflow_patches,  # ⚠️ Always return data (even if invalid)
            'static_attr': static_patches,
            'catchment_padding_mask': padding_mask,
            'num_patches': self.num_patches,
            'patch_size': self.patch_size,
            'start_date': start_date,
            'riverflow_valid': riverflow_valid,  # ⭐ NEW: Validity flag
        }
