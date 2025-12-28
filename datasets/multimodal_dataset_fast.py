"""
优化版Multi-modal Hydrology Dataset - 使用合并后的h5文件

性能提升：
1. 使用合并后的大文件（train/val各一个文件per modality）
2. 支持内存缓存（可选）
3. 减少文件I/O操作10-100倍

对比原版：
- 原版：每个batch需要打开多个h5文件（慢）
- 新版：整个训练期只打开1个h5文件，可选缓存到内存（快）
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


class MultiModalHydroDatasetFast(Dataset):
    """
    优化版Multi-modal Hydrology Dataset
    使用合并后的h5文件，支持内存缓存
    """

    def __init__(
        self,
        # 合并后的h5文件路径
        merged_data_dir: str,
        # Vector modality data (pre-loaded)
        evap_data: np.ndarray,  # [num_catchments, num_days]
        riverflow_data: np.ndarray,  # [num_catchments, num_days]
        # Static attributes
        static_attr_file: str,
        static_attr_vars: List[str],
        # Time range
        start_date: str,  # 'YYYY-MM-DD'
        end_date: str,  # 'YYYY-MM-DD'
        # Sampling parameters
        max_sequence_length: int = 90,
        stride: int = 30,
        # Catchment configuration
        catchment_ids: Optional[np.ndarray] = None,
        # Normalization
        stats_cache_path: Optional[str] = None,
        land_mask_path: Optional[str] = None,
        # Other
        split: str = 'train',  # 'train'/'val'/'test'
        # Performance optimization
        cache_to_memory: bool = True,  # 是否缓存到内存
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.stride = stride
        self.split = split
        self.static_attr_vars = static_attr_vars
        self.cache_to_memory = cache_to_memory

        # Generate date list
        self.date_list = self._generate_date_list(start_date, end_date)
        self.num_days = len(self.date_list)
        print(f"Date range: {self.date_list[0]} to {self.date_list[-1]} ({self.num_days} days)")

        # Store catchment IDs and vector data
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

        self.evap_data = evap_data
        self.riverflow_data = riverflow_data

        # Load merged h5 files
        self.image_data = self._load_merged_h5_files(merged_data_dir, start_date, end_date, split)

        # Load static attributes
        self.static_attrs = self._load_static_attributes(
            static_attr_file, self.catchment_ids, static_attr_vars
        )

        # Load or compute normalization stats
        if stats_cache_path and os.path.exists(stats_cache_path):
            print(f"Loading normalization stats from {stats_cache_path}")
            self.stats = torch.load(stats_cache_path)
        else:
            print("Computing normalization stats...")
            if land_mask_path is None:
                raise ValueError("land_mask_path required when computing stats")
            self.stats = self._compute_all_stats(land_mask_path)

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

    def _load_merged_h5_files(
        self,
        merged_data_dir: str,
        start_date: str,
        end_date: str,
        split: str
    ) -> Dict[str, np.ndarray]:
        """
        加载合并后的h5文件

        Args:
            merged_data_dir: 合并文件目录
            start_date: 开始日期
            end_date: 结束日期
            split: 'train' or 'val'

        Returns:
            {
                'precip': [num_days, 290, 180],
                'soil': [num_days, 290, 180],
                'temp': [num_days, 290, 180],
            }
        """
        merged_dir = Path(merged_data_dir)

        modalities = ['precipitation', 'soil_moisture', 'temperature']
        modality_map = {
            'precipitation': 'precip',
            'soil_moisture': 'soil',
            'temperature': 'temp'
        }

        image_data = {}

        print(f"\n{'='*60}")
        print(f"Loading merged h5 files from: {merged_data_dir}")
        print(f"{'='*60}")

        for modality_name in modalities:
            short_name = modality_map[modality_name]

            # 构造文件名
            # 格式: {modality}_{split}_{start_year}_{end_year}.h5
            start_year = start_date[:4]
            end_year = end_date[:4]
            filename = f"{modality_name}_{split}_{start_year}_{end_year}.h5"
            file_path = merged_dir / filename

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Merged h5 file not found: {file_path}\n"
                    f"Please run: python utils/merge_h5_files.py"
                )

            print(f"\nLoading {modality_name}...")
            print(f"  File: {file_path}")

            with h5py.File(file_path, 'r') as f:
                if self.cache_to_memory:
                    # 缓存到内存（最快）
                    print(f"  Caching to memory...")
                    data = f['data'][:]  # 读取全部数据到内存
                    print(f"  ✓ Cached: {data.shape}, {data.nbytes / (1024**2):.2f} MB")
                else:
                    # 保持h5py dataset引用（按需读取）
                    print(f"  Using on-demand loading (slower)")
                    data = f['data']  # 注意：这里不能离开with块

                image_data[short_name] = data

        if not self.cache_to_memory:
            print("\n⚠️  Warning: cache_to_memory=False may be slow!")
            print("   Consider setting cache_to_memory=True for better performance")

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

    def _compute_all_stats(self, land_mask_path: str) -> Dict:
        """Compute all normalization statistics"""
        land_mask = torch.load(land_mask_path)
        stats = {'land_mask': land_mask}

        # 1. Image modality stats
        print("Computing image statistics (land pixels only)...")
        for modality in ['precip', 'soil', 'temp']:
            img_stats = self._compute_image_stats(modality, land_mask, num_samples=1000)
            stats[f'{modality}_mean'] = img_stats['mean']
            stats[f'{modality}_std'] = img_stats['std']
            print(f"  {modality}: mean={img_stats['mean'].item():.4f}, std={img_stats['std'].item():.4f}")

        # 2. Vector modality stats
        print("Computing vector statistics (per-catchment)...")
        vec_stats = self._compute_vector_stats()
        stats.update(vec_stats)

        # 3. Static attribute stats
        print("Computing static attribute statistics...")
        static_stats = self._compute_static_stats()
        stats['static_mean'] = static_stats['mean']
        stats['static_std'] = static_stats['std']

        return stats

    def _compute_image_stats(
        self,
        modality: str,
        land_mask: torch.Tensor,
        num_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Compute statistics for image modality"""
        all_land_values = []

        sample_indices = np.random.choice(
            self.num_days,
            min(num_samples, self.num_days),
            replace=False
        )

        for idx in sample_indices:
            img = self.image_data[modality][idx]  # [290, 180]
            land_values = img[land_mask.numpy() == 1]
            all_land_values.append(land_values)

        all_land_values = np.concatenate(all_land_values)
        mean = torch.tensor(all_land_values.mean())
        std = torch.tensor(all_land_values.std())

        return {'mean': mean, 'std': std}

    def _compute_vector_stats(self) -> Dict[str, torch.Tensor]:
        """Compute per-catchment statistics for vector modalities"""
        evap_means, evap_stds = [], []
        riverflow_means, riverflow_stds = [], []

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

    def _build_valid_samples(self) -> List[Tuple[int, int]]:
        """Build valid sample indices"""
        valid_samples = []

        for catch_idx in range(self.num_catchments):
            for day_idx in range(0, self.num_days - self.max_sequence_length + 1, self.stride):
                # Check riverflow
                window_riverflow = self.riverflow_data[
                    catch_idx,
                    day_idx:day_idx + self.max_sequence_length
                ]

                if np.isnan(window_riverflow).any():
                    continue

                valid_samples.append((catch_idx, day_idx))

        return valid_samples

    def _normalize_image(
        self,
        img_seq: np.ndarray,
        modality: str
    ) -> np.ndarray:
        """Normalize image sequence"""
        img_norm = img_seq.copy()
        land_mask = self.stats['land_mask'].numpy()

        mean = self.stats[f'{modality}_mean'].item()
        std = self.stats[f'{modality}_std'].item()

        for t in range(img_seq.shape[0]):
            img_norm[t][land_mask == 1] = (
                (img_seq[t][land_mask == 1] - mean) / (std + 1e-8)
            )
            img_norm[t][land_mask == 0] = 0.0

        return img_norm

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Return a sample

        Returns:
            {
                'precip': [T, 290, 180],
                'soil': [T, 290, 180],
                'temp': [T, 290, 180],
                'evap': [T],
                'riverflow': [T],
                'static_attr': [num_features],
                'catchment_idx': int,
                'catchment_id': int,
                'start_date': datetime,
            }
        """
        catchment_idx, start_day_idx = self.valid_samples[idx]
        end_day_idx = start_day_idx + self.max_sequence_length

        # Load image sequences (直接切片，超快！)
        precip_seq = self.image_data['precip'][start_day_idx:end_day_idx]
        soil_seq = self.image_data['soil'][start_day_idx:end_day_idx]
        temp_seq = self.image_data['temp'][start_day_idx:end_day_idx]

        # Normalize
        precip_norm = self._normalize_image(precip_seq, 'precip')
        soil_norm = self._normalize_image(soil_seq, 'soil')
        temp_norm = self._normalize_image(temp_seq, 'temp')

        # Vector data
        evap_seq = self.evap_data[catchment_idx, start_day_idx:end_day_idx]
        riverflow_seq = self.riverflow_data[catchment_idx, start_day_idx:end_day_idx]

        # Normalize vectors
        evap_norm = (evap_seq - self.stats['evap_mean'][catchment_idx].item()) / (
            self.stats['evap_std'][catchment_idx].item() + 1e-8
        )
        riverflow_norm = (riverflow_seq - self.stats['riverflow_mean'][catchment_idx].item()) / (
            self.stats['riverflow_std'][catchment_idx].item() + 1e-8
        )

        # Normalize static attributes
        static_norm = (
            self.static_attrs[catchment_idx] - self.stats['static_mean']
        ) / (self.stats['static_std'] + 1e-8)

        return {
            'precip': precip_norm,
            'soil': soil_norm,
            'temp': temp_norm,
            'evap': evap_norm,
            'riverflow': riverflow_norm,
            'static_attr': static_norm,
            'catchment_idx': catchment_idx,
            'catchment_id': self.catchment_ids[catchment_idx],
            'start_date': self.date_list[start_day_idx],
        }
