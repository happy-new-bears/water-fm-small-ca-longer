"""
创建10×10网格的Spatial Patches

读取catchment经纬度和面积信息，生成patch assignments
"""

import pandas as pd
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.spatial_aggregation import create_grid_patches


def load_catchment_data(catchment_file: str):
    """
    从CAMELS_GB_topographic_attributes.csv加载catchment信息

    Args:
        catchment_file: path to topographic attributes CSV

    Returns:
        catchment_ids: list of gauge IDs
        lons: [num_catchments] longitude tensor
        lats: [num_catchments] latitude tensor
        areas: [num_catchments] area tensor (km²)
    """
    df = pd.read_csv(catchment_file)

    print(f"Loaded {len(df)} catchments from {catchment_file}")
    print(f"Columns: {df.columns.tolist()}")

    # Extract data
    catchment_ids = df['gauge_id'].tolist()
    lons = torch.tensor(df['gauge_lon'].values, dtype=torch.float32)
    lats = torch.tensor(df['gauge_lat'].values, dtype=torch.float32)
    areas = torch.tensor(df['area'].values, dtype=torch.float32)

    # Print statistics
    print(f"\nCatchment Statistics:")
    print(f"  Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")
    print(f"  Latitude range: [{lats.min():.2f}, {lats.max():.2f}]")
    print(f"  Area range: [{areas.min():.2f}, {areas.max():.2f}] km²")
    print(f"  Total area: {areas.sum():.2f} km²")

    return catchment_ids, lons, lats, areas


def create_and_save_patches(
    catchment_file: str,
    output_file: str,
    grid_size: tuple = (10, 10),
    catchment_ids_filter: list = None,
):
    """
    创建spatial patches并保存

    Args:
        catchment_file: path to CAMELS_GB_topographic_attributes.csv
        output_file: output .pt file path
        grid_size: (M, N) grid size (default 10×10)
        catchment_ids_filter: optional list of catchment IDs to use (for filtering to 604)

    Saves:
        A .pt file with:
            - patch_assignments: [num_catchments]
            - catchment_areas: [num_catchments]
            - catchment_ids: list of gauge IDs
            - num_patches: int
            - grid_size: tuple
            - catchment_lons: [num_catchments]
            - catchment_lats: [num_catchments]
    """
    # Load data
    catchment_ids, lons, lats, areas = load_catchment_data(catchment_file)

    # Optional: filter to specific catchment IDs
    if catchment_ids_filter is not None:
        print(f"\nFiltering to {len(catchment_ids_filter)} specified catchments...")

        # Create mask for filtering
        id_to_idx = {cid: idx for idx, cid in enumerate(catchment_ids)}
        filter_mask = [id_to_idx[cid] for cid in catchment_ids_filter if cid in id_to_idx]

        catchment_ids = [catchment_ids[i] for i in filter_mask]
        lons = lons[filter_mask]
        lats = lats[filter_mask]
        areas = areas[filter_mask]

        print(f"After filtering: {len(catchment_ids)} catchments")

    num_catchments = len(catchment_ids)

    # Create grid patches
    print(f"\nCreating {grid_size[0]}×{grid_size[1]} grid patches...")
    patch_assignments, num_patches = create_grid_patches(
        lons, lats, areas, grid_size=grid_size
    )

    # Statistics
    unique_patches = torch.unique(patch_assignments)
    print(f"\n✓ Created {num_patches} total patches")
    print(f"  Non-empty patches: {len(unique_patches)}")
    print(f"  Empty patches: {num_patches - len(unique_patches)}")

    # Patch size distribution
    patch_sizes = [(patch_assignments == i).sum().item() for i in range(num_patches)]
    non_zero_sizes = [s for s in patch_sizes if s > 0]
    print(f"\n  Patch size distribution:")
    print(f"    Min: {min(non_zero_sizes)}")
    print(f"    Max: {max(non_zero_sizes)}")
    print(f"    Mean: {sum(non_zero_sizes) / len(non_zero_sizes):.1f}")
    print(f"    Total catchments: {sum(patch_sizes)}")

    # Save
    save_dict = {
        'patch_assignments': patch_assignments,
        'catchment_areas': areas,
        'catchment_ids': catchment_ids,
        'num_catchments': num_catchments,
        'num_patches': num_patches,
        'grid_size': grid_size,
        'catchment_lons': lons,
        'catchment_lats': lats,
    }

    torch.save(save_dict, output_file)
    print(f"\n✓ Saved spatial patches to: {output_file}")

    return save_dict


if __name__ == '__main__':
    """
    运行脚本生成10×10网格patches
    """
    print("=" * 80)
    print("Creating 10×10 Spatial Patches for Catchments")
    print("=" * 80)

    # Paths
    catchment_file = '/Users/transformer/Desktop/water_data/new_version/Catchment_attributes/CAMELS_GB_topographic_attributes.csv'
    output_file = '/Users/transformer/Desktop/water_code/water_fm_small_ca/data/spatial_patches_10x10.pt'

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Option 1: Use all catchments (671 from file)
    print("\n📍 Option 1: Using ALL catchments from file")
    result = create_and_save_patches(
        catchment_file=catchment_file,
        output_file=output_file,
        grid_size=(10, 10),
        catchment_ids_filter=None,  # Use all
    )

    # Option 2: If you have a specific list of 604 catchment IDs to use
    # Uncomment and provide the list if needed
    """
    print("\n" + "=" * 80)
    print("📍 Option 2: Using FILTERED 604 catchments")
    print("=" * 80)

    # Load your 604 catchment IDs
    # Example: catchment_ids_604 = torch.load('path/to/your/604_catchment_ids.pt')
    # Or read from your dataset

    catchment_ids_604 = [...]  # Your list of 604 IDs

    output_file_filtered = '/Users/transformer/Desktop/water_code/water_fm_small_ca/data/spatial_patches_10x10_filtered.pt'

    result_filtered = create_and_save_patches(
        catchment_file=catchment_file,
        output_file=output_file_filtered,
        grid_size=(10, 10),
        catchment_ids_filter=catchment_ids_604,
    )
    """

    print("\n" + "=" * 80)
    print("✓✓✓ Spatial Patches Created Successfully ✓✓✓")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Update configs/mae_config.py:")
    print(f"   use_spatial_agg = True")
    print(f"   spatial_patches_file = '{output_file}'")
    print(f"2. Train model with spatial aggregation enabled")
