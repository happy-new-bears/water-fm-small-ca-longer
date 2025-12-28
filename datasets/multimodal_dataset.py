"""
Multi-modal Hydrology Dataset for MAE-style pretraining
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
from collections import defaultdict


class MultiModalHydroDataset(Dataset):
    """
    Multi-modal Hydrology Dataset supporting MAE-style pretraining.

    Handles 5 modalities:
    - 3 image modalities: precipitation, soil_moisture, temperature
    - 2 vector modalities: evaporation, riverflow (patchified)
    - Static attributes: catchment characteristics

    Features:
    - Per-catchment normalization for vectors
    - Land-only normalization for images
    - Spatial patchify for vector modalities (patch_size=8)
    - Each sample contains full-field data (all catchments + full image)
    - Samples are indexed by time windows only (no per-catchment duplication)
    """

    def __init__(
        self,
        # Image modality paths (legacy mode)
        precip_dir: Optional[str] = None,
        soil_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
        # Merged h5 files (optimized mode) - takes priority if provided
        precip_h5: Optional[str] = None,
        soil_h5: Optional[str] = None,
        temp_h5: Optional[str] = None,
        # Vector modality data (pre-loaded)
        evap_data: np.ndarray = None,  # [num_catchments, num_days]
        riverflow_data: np.ndarray = None,  # [num_catchments, num_days]
        # Static attributes
        static_attr_file: str = None,
        static_attr_vars: List[str] = None,
        # Time range
        start_date: str = None,  # 'YYYY-MM-DD'
        end_date: str = None,  # 'YYYY-MM-DD'
        # Sampling parameters
        max_sequence_length: int = 90,  # Maximum sequence length to return
        stride: int = 30,  # Stride for sliding window sampling (days)
        # Catchment configuration
        catchment_ids: Optional[np.ndarray] = None,  # None = use all catchments
        # Normalization
        stats_cache_path: Optional[str] = None,
        land_mask_path: Optional[str] = None,
        # Other
        split: str = 'train',  # 'train'/'val'/'test'
        # Performance optimization
        cache_to_memory: bool = False,  # Cache merged h5 data to memory (fastest)
    ):
        super().__init__()

        # Determine mode: merged h5 (optimized) or directory mode (legacy)
        self.use_merged = (precip_h5 is not None and soil_h5 is not None and temp_h5 is not None)
        self.cache_to_memory = cache_to_memory

        if self.use_merged:
            print(f"\n{'='*60}")
            print("Using OPTIMIZED mode (merged h5 files)")
            print(f"{'='*60}")
            self.precip_h5 = precip_h5
            self.soil_h5 = soil_h5
            self.temp_h5 = temp_h5
            self.image_data = {}  # Will store loaded data
        else:
            print(f"\n{'='*60}")
            print("Using LEGACY mode (directory-based h5 files)")
            print(f"{'='*60}")
            if precip_dir is None or soil_dir is None or temp_dir is None:
                raise ValueError(
                    "Either provide (precip_h5, soil_h5, temp_h5) for optimized mode, "
                    "or (precip_dir, soil_dir, temp_dir) for legacy mode"
                )
            self.precip_dir = Path(precip_dir)
            self.soil_dir = Path(soil_dir)
            self.temp_dir = Path(temp_dir)

        self.max_sequence_length = max_sequence_length
        self.stride = stride
        self.split = split

        # Static attributes
        self.static_attr_vars = static_attr_vars

        # Generate date list
        self.date_list = self._generate_date_list(start_date, end_date)
        self.num_days = len(self.date_list)
        print(f"Date range: {self.date_list[0]} to {self.date_list[-1]} ({self.num_days} days)")

        # Store catchment IDs and data
        # The evap_data and riverflow_data are already [num_catchments, num_days]
        # catchment_ids should correspond to the rows in these arrays
        self.num_catchments = evap_data.shape[0]
        if catchment_ids is not None:
            if len(catchment_ids) != self.num_catchments:
                raise ValueError(
                    f"Length of catchment_ids ({len(catchment_ids)}) must match "
                    f"first dimension of data ({self.num_catchments})"
                )
            self.catchment_ids = catchment_ids
        else:
            # Create default IDs if not provided
            self.catchment_ids = np.arange(self.num_catchments)

        # Store vector data directly (no subsetting needed)
        self.evap_data = evap_data  # [num_catchments, num_days]
        self.riverflow_data = riverflow_data  # [num_catchments, num_days]

        # Load image data
        if self.use_merged:
            self._load_merged_h5()
        else:
            # Scan h5 files (legacy mode)
            self.h5_file_map = self._scan_h5_files()

        # Load static attributes
        self.static_attrs = self._load_static_attributes(
            static_attr_file, self.catchment_ids, static_attr_vars
        )

        # Load or compute normalization stats
        if stats_cache_path and os.path.exists(stats_cache_path):
            print(f"Loading normalization stats from {stats_cache_path}")
            self.stats = torch.load(stats_cache_path)
        else:
            print("Computing normalization stats (this may take a while)...")
            if land_mask_path is None:
                raise ValueError("land_mask_path required when computing stats")
            self.stats = self._compute_all_stats(land_mask_path)

            # Save stats
            if stats_cache_path:
                os.makedirs(os.path.dirname(stats_cache_path), exist_ok=True)
                torch.save(self.stats, stats_cache_path)
                print(f"Saved normalization stats to {stats_cache_path}")

        # Build valid sample indices
        self.valid_samples = self._build_valid_samples()

        print(f"Dataset initialized: {len(self.valid_samples)} valid samples, "
              f"{self.num_catchments} catchments, {self.num_days} days")

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

    def _scan_h5_files(self) -> Dict[str, Dict[str, Path]]:
        """
        Scan h5 files for three image modalities

        Returns:
            {
                'precip': {'2020-01': Path(...), '2020-02': Path(...), ...},
                'soil': {...},
                'temp': {...}
            }
        """
        file_map = {}

        for modality, dir_path in [
            ('precip', self.precip_dir),
            ('soil', self.soil_dir),
            ('temp', self.temp_dir)
        ]:
            file_map[modality] = {}

            # Scan h5 files
            for h5_file in sorted(dir_path.glob('*.h5')):
                # Extract year-month from filename
                # Expected format: {modality}_YYYY_MM.h5
                stem = h5_file.stem

                # Try to parse year and month
                parts = stem.split('_')
                if len(parts) >= 3:
                    try:
                        year = parts[-2]
                        month = parts[-1]
                        year_month = f"{year}-{month}"
                        file_map[modality][year_month] = h5_file
                    except:
                        print(f"Warning: Cannot parse {h5_file.name}")
                        continue

        # Verify we found files
        for modality in ['precip', 'soil', 'temp']:
            if not file_map[modality]:
                raise ValueError(f"No h5 files found for {modality} in {getattr(self, f'{modality}_dir')}")

        print(f"Found h5 files: precip={len(file_map['precip'])}, "
              f"soil={len(file_map['soil'])}, temp={len(file_map['temp'])}")

        return file_map

    def _load_merged_h5(self):
        """
        Load merged h5 files (optimized mode)

        Loads data from pre-merged h5 files, optionally caching to memory
        """
        modality_files = {
            'precip': self.precip_h5,
            'soil': self.soil_h5,
            'temp': self.temp_h5,
        }

        for modality, h5_path in modality_files.items():
            if not os.path.exists(h5_path):
                raise FileNotFoundError(
                    f"Merged h5 file not found: {h5_path}\n"
                    f"Please ensure the merged files exist or use legacy mode with directory paths."
                )

            print(f"\nLoading {modality} from {h5_path}...")

            with h5py.File(h5_path, 'r') as f:
                if self.cache_to_memory:
                    # Cache to memory (fastest)
                    print(f"  Caching to memory...")
                    data = f['data'][:]  # Load all data to RAM
                    print(f"  ✓ Cached: {data.shape}, {data.nbytes / (1024**2):.2f} MB")
                    self.image_data[modality] = data
                else:
                    # Keep h5py file handle open (slower but uses less memory)
                    # Re-open file and keep it open
                    self._h5_handles = getattr(self, '_h5_handles', {})
                    self._h5_handles[modality] = h5py.File(h5_path, 'r')
                    self.image_data[modality] = self._h5_handles[modality]['data']
                    print(f"  ✓ File opened (on-demand loading)")

        if self.cache_to_memory:
            print(f"\n✓ All image data cached to memory (fastest mode)")
        else:
            print(f"\n✓ Using on-demand loading (slower but less memory)")
            print(f"  💡 Tip: Set cache_to_memory=True for 10-100x speedup")

    def _load_static_attributes(
        self,
        file_path: str,
        catchment_ids: np.ndarray,
        vars_to_use: List[str]
    ) -> torch.Tensor:
        """
        Load static attributes from CSV and align with catchment IDs

        Returns:
            torch.Tensor: [num_catchments, num_features]
        """
        # Read CSV
        df = pd.read_csv(file_path)

        # Use 'id' column for catchment IDs
        id_col = 'id'
        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' not found in {file_path}. Available: {list(df.columns)}")

        # Drop duplicates
        df = df.drop_duplicates(subset=[id_col], keep='first')

        # Filter catchments
        df_filtered = df[df[id_col].isin(catchment_ids)]

        # Check for missing catchments
        missing = set(catchment_ids) - set(df_filtered[id_col])
        if missing:
            print(f"Warning: {len(missing)} catchments missing in attributes")

        # Order by catchment IDs
        df_ordered = df_filtered.set_index(id_col).loc[catchment_ids].reset_index()

        # Extract attribute columns
        try:
            attrs = df_ordered[vars_to_use].values.astype(np.float32)
        except KeyError as e:
            raise KeyError(f"Attributes {e} not found. Available: {list(df.columns)}")

        # Handle NaN
        if np.isnan(attrs).any():
            print("Warning: NaN in static attributes, filling with column means")
            col_means = np.nanmean(attrs, axis=0)
            for i in range(attrs.shape[1]):
                attrs[np.isnan(attrs[:, i]), i] = col_means[i]

        return torch.from_numpy(attrs)

    def _compute_all_stats(self, land_mask_path: str) -> Dict:
        """Compute all normalization statistics"""
        # Load land mask
        land_mask = torch.load(land_mask_path)
        stats = {'land_mask': land_mask}

        # 1. Compute image modality stats (land pixels only)
        print("Computing image statistics (land pixels only)...")
        for modality in ['precip', 'soil', 'temp']:
            img_stats = self._compute_image_stats(modality, land_mask, num_samples=1000)
            stats[f'{modality}_mean'] = img_stats['mean']
            stats[f'{modality}_std'] = img_stats['std']
            print(f"  {modality}: mean={img_stats['mean'].item():.4f}, std={img_stats['std'].item():.4f}")

        # 2. Compute vector modality stats (per-catchment)
        print("Computing vector statistics (per-catchment)...")
        vec_stats = self._compute_vector_stats()
        stats['evap_mean'] = vec_stats['evap_mean']
        stats['evap_std'] = vec_stats['evap_std']
        stats['riverflow_mean'] = vec_stats['riverflow_mean']
        stats['riverflow_std'] = vec_stats['riverflow_std']

        # 3. Compute static attribute stats (global)
        print("Computing static attribute statistics (global)...")
        static_stats = self._compute_static_stats()
        stats['static_mean'] = static_stats['mean']
        stats['static_std'] = static_stats['std']

        stats['num_samples_used'] = 1000

        return stats

    def _compute_image_stats(
        self,
        modality: str,
        land_mask: torch.Tensor,
        num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Compute statistics for image modality (land pixels only)"""
        all_land_values = []

        # Sample random dates
        sample_indices = np.random.choice(len(self.date_list), min(num_samples, len(self.date_list)), replace=False)

        for idx in sample_indices:
            date = self.date_list[idx]
            try:
                # Load single image
                img = self._load_single_image(modality, date)  # [290, 180]

                # Extract land pixel values
                land_values = img[land_mask == 1]
                all_land_values.append(land_values)
            except:
                continue

        # Compute statistics
        all_land_values = np.concatenate(all_land_values)
        mean = torch.tensor(all_land_values.mean())
        std = torch.tensor(all_land_values.std())

        return {'mean': mean, 'std': std}

    def _compute_vector_stats(self) -> Dict[str, torch.Tensor]:
        """Compute per-catchment statistics for vector modalities"""
        evap_means = []
        evap_stds = []
        riverflow_means = []
        riverflow_stds = []

        for catch_idx in range(self.num_catchments):
            # Evaporation
            evap_data = self.evap_data[catch_idx]
            valid_evap = evap_data[~np.isnan(evap_data)]
            if len(valid_evap) > 0:
                evap_means.append(valid_evap.mean())
                evap_stds.append(valid_evap.std())
            else:
                evap_means.append(0.0)
                evap_stds.append(1.0)

            # Riverflow
            river_data = self.riverflow_data[catch_idx]
            valid_river = river_data[~np.isnan(river_data)]
            if len(valid_river) > 0:
                riverflow_means.append(valid_river.mean())
                riverflow_stds.append(valid_river.std())
            else:
                riverflow_means.append(0.0)
                riverflow_stds.append(1.0)

        return {
            'evap_mean': torch.tensor(evap_means, dtype=torch.float32),
            'evap_std': torch.tensor(evap_stds, dtype=torch.float32),
            'riverflow_mean': torch.tensor(riverflow_means, dtype=torch.float32),
            'riverflow_std': torch.tensor(riverflow_stds, dtype=torch.float32),
        }

    def _compute_static_stats(self) -> Dict[str, torch.Tensor]:
        """Compute global statistics for static attributes"""
        return {
            'mean': self.static_attrs.mean(dim=0),
            'std': self.static_attrs.std(dim=0),
        }

    def _build_valid_samples(self) -> List[int]:
        """
        Build valid sample indices (time windows only)

        Rules:
        1. Must have sufficient length (max_sequence_length days)
        2. Use stride for sliding window (default: 30 days)

        Returns:
            List of day indices for valid time windows
        """
        valid_samples = []

        for day_idx in range(0, self.num_days - self.max_sequence_length + 1, self.stride):
            valid_samples.append(day_idx)

        return valid_samples

    def _load_single_image(self, modality: str, date: datetime) -> np.ndarray:
        """Load a single image for a given date"""
        year_month = date.strftime('%Y-%m')
        h5_path = self.h5_file_map[modality][year_month]

        with h5py.File(h5_path, 'r') as f:
            date_key = date.strftime('%Y-%m-%d')
            img = f[date_key][:]

        return img

    def _load_image_sequence(
        self,
        modality: str,
        date_range: List[datetime]
    ) -> np.ndarray:
        """
        Load image sequence for specified dates

        Args:
            modality: 'precip', 'soil', or 'temp'
            date_range: List of dates

        Returns:
            np.ndarray: [T, 290, 180]
        """
        T = len(date_range)
        result = np.zeros((T, 290, 180), dtype=np.float32)

        # Group by month to reduce file opening
        monthly_groups = defaultdict(list)
        for i, date in enumerate(date_range):
            year_month = date.strftime('%Y-%m')
            monthly_groups[year_month].append((i, date))

        # Load month by month
        for year_month, date_indices in monthly_groups.items():
            h5_path = self.h5_file_map[modality][year_month]

            with h5py.File(h5_path, 'r') as f:
                for local_idx, date in date_indices:
                    date_key = date.strftime('%Y-%m-%d')
                    result[local_idx] = f[date_key][:]

        return result

    def _normalize_image(
        self,
        img_seq: np.ndarray,  # [T, 290, 180]
        modality: str
    ) -> np.ndarray:
        """Normalize image sequence (land pixels only)"""
        img_norm = img_seq.copy()
        land_mask = self.stats['land_mask'].numpy()  # [290, 180]

        mean = self.stats[f'{modality}_mean'].item()
        std = self.stats[f'{modality}_std'].item()

        # Normalize each timestep
        for t in range(img_seq.shape[0]):
            # Only normalize land pixels
            img_norm[t][land_mask == 1] = (
                (img_seq[t][land_mask == 1] - mean) / (std + 1e-8)
            )
            # Ocean pixels set to 0
            img_norm[t][land_mask == 0] = 0.0

        return img_norm

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __del__(self):
        """Clean up h5 file handles if they were kept open"""
        if hasattr(self, '_h5_handles'):
            for modality, handle in self._h5_handles.items():
                try:
                    handle.close()
                except:
                    pass

    def __getitem__(self, idx: int) -> Dict:
        """
        Return a sample (max_sequence_length days) containing all catchments

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

        # Get date range
        date_range = self.date_list[start_day_idx:end_day_idx]

        # Load image sequences
        if self.use_merged:
            # Optimized: Direct array slicing (10-100x faster!)
            precip_seq = self.image_data['precip'][start_day_idx:end_day_idx]
            soil_seq = self.image_data['soil'][start_day_idx:end_day_idx]
            temp_seq = self.image_data['temp'][start_day_idx:end_day_idx]
        else:
            # Legacy: Load from multiple h5 files
            precip_seq = self._load_image_sequence('precip', date_range)
            soil_seq = self._load_image_sequence('soil', date_range)
            temp_seq = self._load_image_sequence('temp', date_range)

        # Normalize images
        precip_norm = self._normalize_image(precip_seq, 'precip')
        soil_norm = self._normalize_image(soil_seq, 'soil')
        temp_norm = self._normalize_image(temp_seq, 'temp')

        # Get vector sequences for ALL catchments (for spatial patchify)
        # Shape: [num_catchments, time_steps]
        evap_seq_all = self.evap_data[:, start_day_idx:end_day_idx]  # [C, T]
        riverflow_seq_all = self.riverflow_data[:, start_day_idx:end_day_idx]  # [C, T]

        # Normalize vectors (per-catchment)
        evap_norm_list = []
        riverflow_norm_list = []
        for c_idx in range(self.num_catchments):
            evap_norm_c = (evap_seq_all[c_idx] - self.stats['evap_mean'][c_idx].item()) / (
                self.stats['evap_std'][c_idx].item() + 1e-8
            )
            riverflow_norm_c = (riverflow_seq_all[c_idx] - self.stats['riverflow_mean'][c_idx].item()) / (
                self.stats['riverflow_std'][c_idx].item() + 1e-8
            )
            evap_norm_list.append(evap_norm_c)
            riverflow_norm_list.append(riverflow_norm_c)

        # Stack: [num_catchments, time_steps]
        evap_norm_all = np.stack(evap_norm_list, axis=0)
        riverflow_norm_all = np.stack(riverflow_norm_list, axis=0)

        # Patchify: [num_catchments, time_steps] -> [num_patches, patch_size, time_steps]
        patch_size = 8
        num_catchments = evap_norm_all.shape[0]
        num_patches = (num_catchments + patch_size - 1) // patch_size  # Ceiling division

        # Pad to make divisible by patch_size
        if num_catchments % patch_size != 0:
            pad_size = num_patches * patch_size - num_catchments
            evap_pad = np.zeros((pad_size, evap_norm_all.shape[1]))
            riverflow_pad = np.zeros((pad_size, riverflow_norm_all.shape[1]))
            evap_norm_all = np.concatenate([evap_norm_all, evap_pad], axis=0)
            riverflow_norm_all = np.concatenate([riverflow_norm_all, riverflow_pad], axis=0)

        # Reshape to patches: [num_patches, patch_size, time_steps]
        evap_patches = evap_norm_all.reshape(num_patches, patch_size, -1)
        riverflow_patches = riverflow_norm_all.reshape(num_patches, patch_size, -1)

        # Normalize static attributes for ALL catchments (for patchify)
        # Shape: [num_catchments, stat_dim]
        static_norm_all_list = []
        for c_idx in range(self.num_catchments):
            static_norm_c = (
                self.static_attrs[c_idx] - self.stats['static_mean']
            ) / (self.stats['static_std'] + 1e-8)
            static_norm_all_list.append(static_norm_c)

        static_norm_all = torch.stack(static_norm_all_list, dim=0)  # [num_catchments, stat_dim]

        # Pad static attributes if needed
        if num_catchments % patch_size != 0:
            pad_size = num_patches * patch_size - num_catchments
            static_pad = torch.zeros(pad_size, static_norm_all.shape[1])
            static_norm_all = torch.cat([static_norm_all, static_pad], dim=0)

        # Reshape to patches: [num_patches, patch_size, stat_dim]
        static_patches = static_norm_all.reshape(num_patches, patch_size, -1)

        # Create padding mask: [num_patches, patch_size] - True = padding catchment
        padding_mask = torch.zeros(num_patches, patch_size, dtype=torch.bool)
        if num_catchments % patch_size != 0:
            # Mark padded catchments as padding
            last_patch_valid = num_catchments % patch_size
            padding_mask[-1, last_patch_valid:] = True

        return {
            'precip': precip_norm,
            'soil': soil_norm,
            'temp': temp_norm,
            'evap': evap_patches,  # [num_patches, patch_size, time_steps]
            'riverflow': riverflow_patches,  # [num_patches, patch_size, time_steps]
            'static_attr': static_patches,  # [num_patches, patch_size, stat_dim]
            'catchment_padding_mask': padding_mask,  # [num_patches, patch_size]
            'num_patches': num_patches,
            'patch_size': patch_size,
            'start_date': date_range[0],
        }
