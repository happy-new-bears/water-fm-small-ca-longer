"""
Training script for Multi-modal MAE with DeepSpeed

Usage:
    # Single machine, 4 GPUs with default config
    deepspeed --num_gpus=4 train_mae.py

    # With custom config file
    deepspeed --num_gpus=4 train_mae.py --config configs/mae_config_exp1.py

    # Multiple experiments in parallel (different GPUs)
    deepspeed --num_gpus=2 --include localhost:0,1 train_mae.py --config configs/mae_config_exp1.py
    deepspeed --num_gpus=2 --include localhost:2,3 train_mae.py --config configs/mae_config_exp2.py

    # Or use torchrun
    torchrun --nproc_per_node=4 train_mae.py --config configs/mae_config_exp1.py
"""

import os
import sys
import time
import json
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import deepspeed as ds
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from models.multimodal_mae import MultiModalMAE
from datasets.multimodal_dataset_optimized import MultiModalHydroDatasetOptimized
from datasets.data_utils import load_vector_data_from_parquet
from datasets.collate import MultiScaleMaskedCollate
from configs.mae_config import MAEConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-modal MAE Training with DeepSpeed')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (e.g., configs/mae_config_exp1.py). If not provided, uses default config.'
    )
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='Local rank for distributed training (automatically set by DeepSpeed)'
    )
    args = parser.parse_args()
    return args


def load_config_from_file(config_path):
    """
    Load config from a Python file

    Args:
        config_path: Path to config file (e.g., 'configs/mae_config_exp1.py')

    Returns:
        MAEConfig instance from the specified file
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load module from file path
    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Get MAEConfig class from module and instantiate
    if not hasattr(config_module, 'MAEConfig'):
        raise AttributeError(f"Config file {config_path} must define a MAEConfig class")

    config = config_module.MAEConfig()

    return config


def save_config(config, save_dir, filename='config.json'):
    """
    Save configuration to JSON file

    Args:
        config: MAEConfig instance
        save_dir: Directory to save config
        filename: Name of config file (default: config.json)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract all non-callable attributes from config
    config_dict = {}
    for key in dir(config):
        if not key.startswith('_'):
            value = getattr(config, key)
            if not callable(value):
                # Convert non-serializable types
                if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)

    # Save to JSON
    config_path = os.path.join(save_dir, filename)
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    return config_path


def setup_wandb(config, rank, timestamp=None):
    """Initialize WandB (only on rank 0)"""
    if rank == 0 and config.use_wandb:
        import wandb

        # Create run name with timestamp
        run_name = f"mae_run_{timestamp}" if timestamp else None

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config={
                'd_model': config.d_model,
                'decoder_dim': config.decoder_dim,
                'img_encoder_layers': config.img_encoder_layers,
                'vec_encoder_layers': config.vec_encoder_layers,
                'decoder_layers': config.decoder_layers,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'image_mask_ratio': config.image_mask_ratio,
                'vector_mask_ratio': config.vector_mask_ratio,
                'epochs': config.epochs,
                'stride': config.stride,
                'zero_stage': config.zero_stage,
                'use_fp16': config.use_fp16,
                'patch_size': config.patch_size,
                'output_dir': config.output_dir,
            }
        )
        return wandb
    return None


def create_datasets(config, rank):
    """Create train and validation datasets"""

    if rank == 0:
        print("=" * 60)
        print("Loading vector data...")
        print("=" * 60)

    # ⭐ NEW: Define validity ranges for each variable
    # Riverflow is only valid from 1989-01-01 onwards
    train_start = datetime.strptime(config.train_start, '%Y-%m-%d')
    val_end = datetime.strptime(config.val_end, '%Y-%m-%d')
    riverflow_valid_start = datetime(1989, 1, 1)

    variable_validity_ranges = {
        'evaporation': (train_start, val_end),  # Full range 1970-2015
        'discharge_vol': (riverflow_valid_start, val_end),  # Only 1989-2015
    }

    # Load vector data with per-variable validity ranges
    vector_data, time_vec, catchment_ids, var_names = load_vector_data_from_parquet(
        config.vector_file,
        variables=['evaporation', 'discharge_vol'],
        start=train_start,
        end=val_end,  # Load all data
        nan_ratio=0.05,
        variable_validity_ranges=variable_validity_ranges,  # ⭐ NEW
    )

    # Extract modalities
    evap_data = vector_data[:, :, 0].T  # [num_catchments, num_days]
    riverflow_data = vector_data[:, :, 1].T

    if rank == 0:
        print(f"✓ Vector data loaded: {evap_data.shape}")
        print(f"✓ Catchments: {len(catchment_ids)}")

    # Split into train and val by time
    train_end_date = datetime.strptime(config.train_end, '%Y-%m-%d')
    val_start_date = datetime.strptime(config.val_start, '%Y-%m-%d')

    # Find split index
    # Convert time_vec to datetime for comparison (time_vec is numpy.datetime64[D])
    time_vec_datetime = [datetime.strptime(str(d), '%Y-%m-%d') for d in time_vec]

    train_end_idx = None
    for i, date in enumerate(time_vec_datetime):
        if date == train_end_date:
            train_end_idx = i + 1
            break

    if train_end_idx is None:
        raise ValueError(
            f"Train end date {config.train_end} not found in data. "
            f"Available range: {time_vec_datetime[0].strftime('%Y-%m-%d')} to {time_vec_datetime[-1].strftime('%Y-%m-%d')}"
        )

    # Train dataset
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Creating training dataset...")
        print("=" * 60)

    # Check if merged h5 files exist
    use_merged = (
        hasattr(config, 'precip_train_h5') and
        hasattr(config, 'soil_train_h5') and
        hasattr(config, 'temp_train_h5') and
        os.path.exists(config.precip_train_h5)
    )

    if use_merged and rank == 0:
        print("✓ Found merged h5 files - using ULTRA-OPTIMIZED mode with pre-normalization")

    train_dataset = MultiModalHydroDatasetOptimized(
        # Merged h5 files
        precip_h5=config.precip_train_h5 if use_merged else config.precip_train_h5,
        soil_h5=config.soil_train_h5 if use_merged else config.soil_train_h5,
        temp_h5=config.temp_train_h5 if use_merged else config.temp_train_h5,
        # Vector data
        evap_data=evap_data[:, :train_end_idx],
        riverflow_data=riverflow_data[:, :train_end_idx],
        static_attr_file=config.static_attr_file,
        static_attr_vars=config.static_attrs,
        start_date=config.train_start,
        end_date=config.train_end,
        max_sequence_length=config.max_time_steps,
        stride=config.stride,
        catchment_ids=catchment_ids,
        stats_cache_path=config.stats_cache_path,
        land_mask_path=config.land_mask_path,
        patch_size=config.vector_patch_size,  # NEW: Vector patch size
        split='train',
        # Performance optimization
        cache_to_memory=config.cache_images_to_memory if hasattr(config, 'cache_images_to_memory') else True,
    )

    # Validation dataset
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Creating validation dataset...")
        print("=" * 60)

    val_dataset = MultiModalHydroDatasetOptimized(
        # Merged h5 files
        precip_h5=config.precip_val_h5 if use_merged else config.precip_val_h5,
        soil_h5=config.soil_val_h5 if use_merged else config.soil_val_h5,
        temp_h5=config.temp_val_h5 if use_merged else config.temp_val_h5,
        # Vector data
        evap_data=evap_data[:, train_end_idx:],
        riverflow_data=riverflow_data[:, train_end_idx:],
        static_attr_file=config.static_attr_file,
        static_attr_vars=config.static_attrs,
        start_date=config.val_start,
        end_date=config.val_end,
        max_sequence_length=config.max_time_steps,
        stride=config.stride,
        catchment_ids=catchment_ids,
        stats_cache_path=config.stats_cache_path,
        land_mask_path=config.land_mask_path,
        patch_size=config.vector_patch_size,  # NEW: Vector patch size
        split='val',
        # Performance optimization
        cache_to_memory=config.cache_images_to_memory if hasattr(config, 'cache_images_to_memory') else True,
    )

    if rank == 0:
        print(f"\n✓ Train samples: {len(train_dataset)}")
        print(f"✓ Val samples: {len(val_dataset)}")

        # Print normalization statistics
        print("\n" + "="*70)
        print("NORMALIZATION STATISTICS")
        print("="*70)

        stats = train_dataset.stats

        # Image modalities
        print("\n📊 IMAGE MODALITIES:")
        for modality in ['precip', 'soil', 'temp']:
            if f'{modality}_mean' in stats and f'{modality}_std' in stats:
                mean = stats[f'{modality}_mean'].item()
                std = stats[f'{modality}_std'].item()
                print(f"  {modality.upper():12s}: mean = {mean:10.4f}, std = {std:10.4f}")

        # Vector modalities
        print("\n📊 VECTOR MODALITIES (per-catchment statistics):")
        for modality in ['evap', 'riverflow']:
            if f'{modality}_mean' in stats and f'{modality}_std' in stats:
                means = stats[f'{modality}_mean']
                stds = stats[f'{modality}_std']
                print(f"  {modality.upper():12s}:")
                print(f"    Mean range: [{means.min().item():8.4f}, {means.max().item():8.4f}]")
                print(f"    Std range:  [{stds.min().item():8.4f}, {stds.max().item():8.4f}]")
                print(f"    Avg mean:   {means.mean().item():8.4f}")
                print(f"    Avg std:    {stds.mean().item():8.4f}")

        # Static attributes
        if 'static_mean' in stats and 'static_std' in stats:
            print("\n📊 STATIC ATTRIBUTES:")
            static_mean = stats['static_mean']
            static_std = stats['static_std']
            print(f"  Number of attributes: {len(static_mean)}")
            print(f"  Mean range: [{static_mean.min().item():8.4f}, {static_mean.max().item():8.4f}]")
            print(f"  Std range:  [{static_std.min().item():8.4f}, {static_std.max().item():8.4f}]")

        # Land mask info
        if 'land_mask' in stats:
            land_mask = stats['land_mask']
            num_land = (land_mask == 1).sum().item()
            num_ocean = (land_mask == 0).sum().item()
            total = land_mask.numel()
            print("\n📊 LAND MASK:")
            print(f"  Total pixels:  {total:6d}")
            print(f"  Land pixels:   {num_land:6d} ({num_land/total*100:5.2f}%)")
            print(f"  Ocean pixels:  {num_ocean:6d} ({num_ocean/total*100:5.2f}%)")

        print("="*70 + "\n")

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config, rank, world_size):
    """Create distributed dataloaders"""

    # Load land mask for collate function
    valid_patch_indices = None
    if config.land_mask_path:
        land_mask = torch.load(config.land_mask_path)
        # Calculate valid patches
        patch_size = config.patch_size
        num_patches_h = config.image_height // patch_size
        num_patches_w = config.image_width // patch_size

        valid_patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = land_mask[
                    i*patch_size:(i+1)*patch_size,
                    j*patch_size:(j+1)*patch_size
                ]
                land_ratio = patch.sum().item() / (patch_size * patch_size)
                if land_ratio >= config.land_threshold:
                    valid_patches.append(i * num_patches_w + j)

        valid_patch_indices = torch.tensor(valid_patches, dtype=torch.long)

        if rank == 0:
            print(f"\n✓ Valid patches: {len(valid_patches)}/522")

    # Train collate function
    train_collate = MultiScaleMaskedCollate(
        seq_len=config.max_time_steps,
        mask_ratio=config.image_mask_ratio,  # Use config mask ratio
        patch_size=config.patch_size,
        land_mask_path=config.land_mask_path,
        land_threshold=config.land_threshold,
        mask_mode='unified',
        mode='train',
    )

    # Val collate function (same masking as train for fair comparison)
    val_collate = MultiScaleMaskedCollate(
        seq_len=config.max_time_steps,
        mask_ratio=config.image_mask_ratio,  # Use same mask ratio as train
        patch_size=config.patch_size,
        land_mask_path=config.land_mask_path,
        land_threshold=config.land_threshold,
        mask_mode='unified',
        mode='val',
    )

    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=train_collate,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        collate_fn=val_collate,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler, val_sampler, valid_patch_indices


def train_epoch(model, train_loader, epoch, config, rank, world_size, wandb):
    """Train for one epoch"""

    model.train()

    # Start timing
    epoch_start_time = time.time()
    batch_times = []

    epoch_loss = torch.tensor(0.0, device=f'cuda:{rank}')
    epoch_losses = {
        'precip_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
        'soil_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
        'temp_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
        'evap_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
        'riverflow_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
    }
    size = torch.tensor(0, device=f'cuda:{rank}')

    for batch_idx, batch in enumerate(train_loader):
        batch_start_time = time.time()

        # Move batch to device and convert to FP16 if needed
        batch_gpu = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda(rank)
                # Convert float tensors to FP16 when using mixed precision
                if config.use_fp16 and v.dtype == torch.float32:
                    v = v.half()
                batch_gpu[k] = v
            else:
                batch_gpu[k] = v

        # Forward pass
        total_loss, loss_dict = model(batch_gpu)

        # Check for NaN/Inf in loss before backward pass
        if not torch.isfinite(total_loss):
            if rank == 0:
                print(f"\n⚠ WARNING: Non-finite loss detected at epoch {epoch+1}, batch {batch_idx+1}")
                print(f"  total_loss: {total_loss.item()}")
                for key, val in loss_dict.items():
                    print(f"  {key}: {val.item()}")
                print("  Skipping this batch and continuing training...")
            continue  # Skip this batch

        # Backward + step (DeepSpeed API)
        model.backward(total_loss)
        model.step()

        # Accumulate losses
        batch_size = batch_gpu['precip'].size(0)
        epoch_loss += total_loss.item() * batch_size
        for key in epoch_losses.keys():
            epoch_losses[key] += loss_dict[key].item() * batch_size
        size += batch_size

        # Record batch time
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)

        # Log batch metrics
        if batch_idx % config.log_frequency == 0 and rank == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {total_loss.item():.4f} "
                  f"Time: {batch_time:.2f}s")

    # Synchronize losses across processes
    dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(size, op=dist.ReduceOp.SUM)
    for key in epoch_losses.keys():
        dist.all_reduce(epoch_losses[key], op=dist.ReduceOp.SUM)

    # Compute averages
    avg_loss = epoch_loss.item() / size.item()
    avg_losses = {k: v.item() / size.item() for k, v in epoch_losses.items()}

    # Calculate timing statistics
    epoch_time = time.time() - epoch_start_time
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    samples_per_sec = size.item() / epoch_time if epoch_time > 0 else 0

    # Log to WandB (rank 0 only)
    if rank == 0:
        print(f"\nEpoch {epoch+1} Training Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        for key, value in avg_losses.items():
            print(f"  {key}: {value:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f}m)")
        print(f"  Avg Batch Time: {avg_batch_time:.3f}s")
        print(f"  Throughput: {samples_per_sec:.2f} samples/sec")

        if wandb is not None:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': avg_loss,
                'train/epoch_time': epoch_time,
                'train/avg_batch_time': avg_batch_time,
                'train/samples_per_sec': samples_per_sec,
            }
            for key, value in avg_losses.items():
                log_dict[f'train/{key}'] = value
            wandb.log(log_dict)

    return avg_loss


def validate(model, val_loader, epoch, config, rank, world_size, wandb):
    """
    Validation with detailed metrics logging

    Records:
    - Total validation loss
    - Individual modality losses (precip, soil, temp, evap, riverflow)
    - Performance metrics (time, throughput, batch time)
    """

    model.eval()

    # Start timing
    val_start_time = time.time()
    batch_times = []

    # Initialize loss accumulators
    epoch_loss = torch.tensor(0.0, device=f'cuda:{rank}')
    epoch_losses = {
        'precip_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
        'soil_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
        'temp_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
        'evap_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
        'riverflow_loss': torch.tensor(0.0, device=f'cuda:{rank}'),
    }
    size = torch.tensor(0, device=f'cuda:{rank}')

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch_start_time = time.time()

            # Move batch to device and convert to FP16 if needed
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    v = v.cuda(rank)
                    # Convert float tensors to FP16 when using mixed precision
                    if config.use_fp16 and v.dtype == torch.float32:
                        v = v.half()
                    batch_gpu[k] = v
                else:
                    batch_gpu[k] = v

            # Forward pass
            total_loss, loss_dict = model(batch_gpu)

            # Accumulate total loss
            batch_size = batch_gpu['precip'].size(0)
            epoch_loss += total_loss.item() * batch_size

            # Accumulate individual modality losses
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key].item() * batch_size

            size += batch_size

            # Record batch time
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

    # Synchronize losses across all processes
    dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(size, op=dist.ReduceOp.SUM)
    for key in epoch_losses.keys():
        dist.all_reduce(epoch_losses[key], op=dist.ReduceOp.SUM)

    # Calculate averages
    avg_loss = epoch_loss.item() / size.item()
    avg_losses = {key: val.item() / size.item() for key, val in epoch_losses.items()}

    # Calculate timing and performance statistics
    val_time = time.time() - val_start_time
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    samples_per_sec = size.item() / val_time if val_time > 0 else 0

    # Log to console (rank 0 only)
    if rank == 0:
        if epoch == -1:
            print(f"\nBaseline Validation Results (before training):")
        else:
            print(f"\nEpoch {epoch+1} Validation Results:")
        print(f"  Average Loss: {avg_loss:.4f}")
        for key, value in avg_losses.items():
            print(f"  {key}: {value:.4f}")
        print(f"  Validation Time: {val_time:.2f}s ({val_time/60:.2f}m)")
        print(f"  Avg Batch Time: {avg_batch_time:.3f}s")
        print(f"  Throughput: {samples_per_sec:.2f} samples/sec")

        # Log to WandB
        if wandb is not None:
            log_dict = {
                'epoch': 0 if epoch == -1 else epoch + 1,  # Use 0 for baseline
                'val/loss': avg_loss,
                'val/time': val_time,
                'val/avg_batch_time': avg_batch_time,
                'val/samples_per_sec': samples_per_sec,
            }
            # Add individual modality losses
            for key, value in avg_losses.items():
                log_dict[f'val/{key}'] = value

            wandb.log(log_dict)

    return avg_loss


def main():
    """Main training function"""

    # Parse command line arguments
    args = parse_args()

    # Initialize DeepSpeed distributed
    ds.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Multi-modal MAE Pretraining")
        print("=" * 60)
        print(f"Rank {rank} of {world_size}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA devices: {torch.cuda.device_count()}")

    # Load config (custom or default)
    if args.config:
        if rank == 0:
            print(f"\nLoading custom config from: {args.config}")
        config = load_config_from_file(args.config)
    else:
        if rank == 0:
            print(f"\nUsing default config: configs/mae_config.py")
        config = MAEConfig()

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_output_dir = config.output_dir
    config.output_dir = os.path.join(original_output_dir, f'run_{timestamp}')

    if rank == 0:
        print(f"\nOutput directory: {config.output_dir}")

    # Initialize WandB with timestamp
    wandb = setup_wandb(config, rank, timestamp)

    # Create datasets
    train_dataset, val_dataset = create_datasets(config, rank)

    # Print normalization statistics
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Normalization Statistics")
        print("=" * 60)

        stats = train_dataset.stats

        # Image modalities (scalar mean/std)
        print(f"\n📊 IMAGE MODALITIES:")
        for modality in ['precip', 'soil', 'temp']:
            mean = stats[f'{modality}_mean'].item()
            std = stats[f'{modality}_std'].item()
            print(f"\n  {modality.upper()}:")
            print(f"    Mean: {mean:12.6f}")
            print(f"    Std:  {std:12.6f}")

            # Sanity checks
            if mean < 0 and modality == 'precip':
                print(f"    ⚠️  WARNING: Negative mean for precipitation!")
            if std < 1e-6:
                print(f"    ⚠️  WARNING: Very small std ({std:.2e})")

        # Vector modalities (global mean/std)
        print(f"\n📊 VECTOR MODALITIES (Global Normalization):")

        # Evaporation
        evap_mean = stats['evap_mean']  # Scalar
        evap_std = stats['evap_std']    # Scalar
        print(f"\n  EVAP:")
        print(f"    Global mean: {evap_mean.item():10.4f}")
        print(f"    Global std:  {evap_std.item():10.4f}")

        # Riverflow (log-transformed)
        riverflow_log_mean = stats['riverflow_log_mean']  # Scalar
        riverflow_log_std = stats['riverflow_log_std']    # Scalar
        print(f"\n  RIVERFLOW (log-transformed):")
        print(f"    Log mean: {riverflow_log_mean.item():10.4f}")
        print(f"    Log std:  {riverflow_log_std.item():10.4f}")

        # Static attributes
        print(f"\n📊 STATIC ATTRIBUTES ({len(config.static_attrs)} features):")
        static_mean = stats['static_mean']
        static_std = stats['static_std']
        print(f"    Features: {', '.join(config.static_attrs[:3])}...")
        print(f"    Mean range: [{static_mean.min().item():10.4f}, {static_mean.max().item():10.4f}]")
        print(f"    Std range:  [{static_std.min().item():10.4f}, {static_std.max().item():10.4f}]")

        # Land mask info
        land_mask = stats['land_mask']
        num_land = (land_mask == 1).sum().item()
        num_ocean = (land_mask == 0).sum().item()
        total = land_mask.numel()
        print(f"\n📍 LAND MASK:")
        print(f"    Shape: {land_mask.shape}")
        print(f"    Land pixels:  {num_land:6d} ({num_land/total*100:5.2f}%)")
        print(f"    Ocean pixels: {num_ocean:6d} ({num_ocean/total*100:5.2f}%)")

        print(f"\n" + "=" * 60)
        print("✅ All statistics loaded successfully!")
        print("=" * 60)

    # Create dataloaders
    train_loader, val_loader, train_sampler, val_sampler, valid_patch_indices = \
        create_dataloaders(train_dataset, val_dataset, config, rank, world_size)

    # Create model
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Creating model...")
        print("=" * 60)

    model = MultiModalMAE(config, valid_patch_indices)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created")
        print(f"✓ Total parameters: {total_params:,}")

    # DeepSpeed configuration
    ds_config = {
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": config.betas,
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            }
        },
        "zero_optimization": {
            "stage": config.zero_stage,
        },
    }

    # Add gradient clipping if specified in config
    if hasattr(config, 'gradient_clip_norm') and config.gradient_clip_norm > 0:
        ds_config["gradient_clipping"] = config.gradient_clip_norm
        if rank == 0:
            print(f"✓ Gradient clipping enabled: {config.gradient_clip_norm}")

    # Add ZeRO stage-specific config to avoid PyTorch API compatibility issues
    if config.zero_stage > 0:
        ds_config["zero_optimization"].update({
            "contiguous_gradients": False,  # Disable to avoid internal API calls
            "overlap_comm": False,  # Disable to avoid internal API calls
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
        })

    if config.use_fp16:
        # Use config values or defaults
        initial_scale_power = getattr(config, 'initial_scale_power', 16)
        loss_scale_window = getattr(config, 'loss_scale_window', 1000)
        min_loss_scale = getattr(config, 'min_loss_scale', 1)
        hysteresis = getattr(config, 'hysteresis', 2)

        ds_config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,  # Dynamic loss scaling
            "initial_scale_power": initial_scale_power,
            "loss_scale_window": loss_scale_window,
            "hysteresis": hysteresis,
            "min_loss_scale": min_loss_scale,
        }
        if rank == 0:
            print(f"✓ FP16 with dynamic loss scaling:")
            print(f"  - initial_scale: 2^{initial_scale_power} = {2**initial_scale_power}")
            print(f"  - min_loss_scale: {min_loss_scale}")
            print(f"  - loss_scale_window: {loss_scale_window}")
            print(f"  - hysteresis: {hysteresis}")

    # Initialize DeepSpeed
    model, optimizer, _, _ = ds.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    if rank == 0:
        print(f"✓ DeepSpeed initialized (ZeRO stage {config.zero_stage})")

    # Create output directory and save initial config
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        config_path = save_config(config, config.output_dir, 'config.json')
        print(f"✓ Configuration saved to: {config_path}")

    # Training loop
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Starting training...")
        print(f"Training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

    training_start_time = time.time()
    best_val_loss = float('inf')

    # Validate before training (epoch -1) to establish baseline
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Baseline validation (before training)...")
        print("=" * 60)

    initial_val_loss = validate(
        model, val_loader, -1, config, rank, world_size, wandb
    )
    best_val_loss = initial_val_loss

    if rank == 0:
        print(f"Baseline validation loss: {initial_val_loss:.4f}")

    for epoch in range(config.epochs):
        epoch_wall_start = time.time()

        # Set epoch for sampler (ensures different shuffling each epoch)
        train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            model, train_loader, epoch, config, rank, world_size, wandb
        )

        # Validate
        if (epoch + 1) % config.val_frequency == 0:
            val_loss = validate(
                model, val_loader, epoch, config, rank, world_size, wandb
            )

            # Save best model
            if rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save model
                model_path = os.path.join(config.output_dir, 'best_model.pth')
                torch.save(model.module.state_dict(), model_path)

                # Save config with best model
                save_config(config, config.output_dir, 'best_model_config.json')

                print(f"✓ Saved best model (val_loss={val_loss:.4f})")
                print(f"  Model: {model_path}")
                print(f"  Config: {os.path.join(config.output_dir, 'best_model_config.json')}")

        # Calculate epoch wall time and ETA
        epoch_wall_time = time.time() - epoch_wall_start
        elapsed_time = time.time() - training_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = config.epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch+1}/{config.epochs} Summary:")
            print(f"  Wall Time: {epoch_wall_time:.2f}s ({epoch_wall_time/60:.2f}m)")
            print(f"  Total Elapsed: {elapsed_time/3600:.2f}h")
            print(f"  ETA: {eta/3600:.2f}h")
            print("=" * 60)

        # Save checkpoint
        if rank == 0 and (epoch + 1) % config.checkpoint_frequency == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(
                config.output_dir,
                f'ckpt_epoch_{epoch+1}.pth'
            )
            torch.save(model.module.state_dict(), checkpoint_path)

            # Save config with checkpoint
            config_checkpoint_path = save_config(
                config,
                config.output_dir,
                f'ckpt_epoch_{epoch+1}_config.json'
            )

            print(f"✓ Saved checkpoint: {checkpoint_path}")
            print(f"  Config: {config_checkpoint_path}")

            # Remove old checkpoints and their configs
            old_epoch = epoch + 1 - config.keep_last_n_checkpoints * config.checkpoint_frequency
            old_checkpoint = os.path.join(
                config.output_dir,
                f'ckpt_epoch_{old_epoch}.pth'
            )
            old_config = os.path.join(
                config.output_dir,
                f'ckpt_epoch_{old_epoch}_config.json'
            )

            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                print(f"  Removed old checkpoint: {old_checkpoint}")

            if os.path.exists(old_config):
                os.remove(old_config)
                print(f"  Removed old config: {old_config}")

    # Cleanup
    dist.destroy_process_group()

    # Calculate total training time
    total_training_time = time.time() - training_start_time

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Training completed!")
        print(f"Training end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total training time: {total_training_time/3600:.2f}h ({total_training_time/60:.2f}m)")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
