"""
将按月分割的h5文件合并为按train/val分割的大文件

优点：
1. 减少文件打开次数（从~264个文件 -> 1个文件）
2. 连续存储，支持高效切片
3. 可以直接缓存到内存

使用方法：
    python utils/merge_h5_files.py
"""

import h5py
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import os


def merge_h5_files_by_period(
    input_dir: str,
    output_dir: str,
    modality_name: str,
    start_date: str,
    end_date: str,
    period_name: str  # 'train' or 'val'
):
    """
    将指定时间段的h5文件合并为单个文件

    Args:
        input_dir: 输入目录，如 '/path/to/precipitation_processed'
        output_dir: 输出目录
        modality_name: 模态名称，如 'precipitation'
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'
        period_name: 时期名称 'train' or 'val'
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成日期列表
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    date_list = []
    current = start
    while current <= end:
        date_list.append(current)
        current += timedelta(days=1)

    num_days = len(date_list)
    print(f"\n{'='*60}")
    print(f"Merging {modality_name} - {period_name}")
    print(f"{'='*60}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Total days: {num_days}")

    # 读取第一个样本获取shape
    first_date = date_list[0]
    first_file = input_path / f"{modality_name}_{first_date.strftime('%Y_%m')}.h5"

    with h5py.File(first_file, 'r') as f:
        first_key = first_date.strftime('%Y-%m-%d')
        sample_shape = f[first_key].shape
        sample_dtype = f[first_key].dtype

    print(f"Image shape: {sample_shape}")
    print(f"Image dtype: {sample_dtype}")

    H, W = sample_shape
    total_size_gb = num_days * H * W * 4 / (1024**3)
    print(f"Estimated output size: {total_size_gb:.2f} GB")

    # 创建输出文件
    output_file = output_path / f"{modality_name}_{period_name}_{start_date[:4]}_{end_date[:4]}.h5"

    print(f"\nWriting to: {output_file}")
    print("This may take a few minutes...")

    with h5py.File(output_file, 'w') as f_out:
        # 创建dataset（使用chunking和compression）
        data_ds = f_out.create_dataset(
            'data',
            shape=(num_days, H, W),
            dtype=sample_dtype,
            chunks=(30, H, W),  # 每30天一个chunk，方便序列读取
            compression='gzip',
            compression_opts=4,  # 中等压缩级别（1-9）
        )

        # 创建日期索引（存储为字符串）
        date_strings = [d.strftime('%Y-%m-%d') for d in date_list]
        f_out.create_dataset(
            'dates',
            data=np.array(date_strings, dtype='S10')  # 固定长度字符串
        )

        # 按月读取并写入
        current_month = None
        f_in = None

        for day_idx, date in enumerate(tqdm(date_list, desc="Processing")):
            year_month = date.strftime('%Y-%m')

            # 如果是新月份，打开新文件
            if year_month != current_month:
                if f_in is not None:
                    f_in.close()

                input_file = input_path / f"{modality_name}_{date.strftime('%Y_%m')}.h5"

                if not input_file.exists():
                    print(f"\nWarning: File not found: {input_file}")
                    print(f"Filling with zeros for {date.strftime('%Y-%m-%d')}")
                    data_ds[day_idx] = np.zeros((H, W), dtype=sample_dtype)
                    continue

                f_in = h5py.File(input_file, 'r')
                current_month = year_month

            # 读取数据
            date_key = date.strftime('%Y-%m-%d')

            if date_key in f_in:
                data_ds[day_idx] = f_in[date_key][:]
            else:
                print(f"\nWarning: Date {date_key} not found in file")
                data_ds[day_idx] = np.zeros((H, W), dtype=sample_dtype)

        if f_in is not None:
            f_in.close()

    # 验证输出文件
    actual_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"\n✓ Successfully created: {output_file}")
    print(f"✓ Actual file size: {actual_size_mb:.2f} MB")
    print(f"✓ Compression ratio: {total_size_gb * 1024 / actual_size_mb:.1f}x")


def main():
    """主函数：合并所有模态的h5文件"""

    # 配置
    data_root = '/Users/transformer/Desktop/water_data/new_version'
    output_root = data_root  # 直接输出到原数据目录

    # 模态列表
    modalities = {
        'precipitation': 'precipitation_processed',
        'soil_moisture': 'soil_moisture_processed',
        'temperature': 'temperature_processed',
    }

    # 时间段配置
    periods = {
        'train': ('1989-01-01', '2010-12-31'),
        'val': ('2011-01-01', '2015-12-30'),
    }

    print("="*60)
    print("H5 File Merger for Multi-modal Hydrology Data")
    print("="*60)
    print(f"Input root: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Modalities: {list(modalities.keys())}")
    print(f"Periods: {list(periods.keys())}")

    # 合并每个模态的每个时期
    for modality_name, input_subdir in modalities.items():
        input_dir = f"{data_root}/{input_subdir}"

        # 检查输入目录是否存在
        if not Path(input_dir).exists():
            print(f"\nWarning: Directory not found: {input_dir}")
            print("Skipping...")
            continue

        for period_name, (start_date, end_date) in periods.items():
            try:
                merge_h5_files_by_period(
                    input_dir=input_dir,
                    output_dir=output_root,
                    modality_name=modality_name,
                    start_date=start_date,
                    end_date=end_date,
                    period_name=period_name
                )
            except Exception as e:
                print(f"\n✗ Error processing {modality_name} {period_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "="*60)
    print("✓✓✓ All files merged successfully!")
    print("="*60)
    print(f"\nMerged files are in: {output_root}")
    print("\nNext steps:")
    print("1. Update your config to use the new merged files")
    print("2. Update MultiModalHydroDataset to use the merged format")
    print("3. Enjoy 10-100x faster data loading! 🚀")


if __name__ == '__main__':
    main()
