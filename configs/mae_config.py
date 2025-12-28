"""
Configuration for Multi-modal MAE pretraining
"""


class MAEConfig:
    """
    Configuration for Multi-modal MAE model and training
    """

    # ========== Model Architecture ==========
    # Encoder settings
    d_model = 256  # Embedding dimension for encoders
    img_encoder_layers = 6  # Number of transformer layers in image encoders
    vec_encoder_layers = 4  # Number of transformer layers in vector encoders with FiLM
    nhead = 8  # Number of attention heads
    dropout = 0.1  # Dropout rate

    # Decoder settings
    decoder_dim = 128  # Embedding dimension for decoders (smaller than encoder)
    decoder_layers = 4  # Number of transformer layers in decoders

    # ========== CrossMAE Configuration ==========
    # Decoder type
    use_cross_attn = True  # Use CrossAttention decoder instead of Self-Attention (CrossMAE)
    decoder_self_attn = False  # Add self-attention in decoder (optional, usually False)

    # Weighted Feature Maps (Phase 2 - optional)
    use_weighted_fm = False  # Enable WeightedFeatureMaps (多层encoder features)
    use_fm_layers = None  # Which encoder layers to use: [0, 2, 4, 5] or None (all)
    use_input = False  # Include input as layer 0

    # Image patch settings
    patch_size = 10  # Patch size for images (10x10)
    image_height = 290  # Image height
    image_width = 180  # Image width

    # Vector patch settings (for spatial aggregation)
    vector_patch_size = 8  # Number of catchments per patch
    num_catchments = 604  # Total number of catchments

    # Sequence settings
    max_time_steps = 30  # Maximum sequence length (days)
    static_attr_dim = 11  # Number of static attributes

    # ========== Data Configuration ==========
    # Data root directory
    data_root = '/Users/transformer/Desktop/water_data/new_version'
    #data_root = '../../new_version'

    # Data paths (relative to data_root)
    precip_dir = f'{data_root}/precipitation_processed'
    soil_dir = f'{data_root}/soil_moisture_processed'
    temp_dir = f'{data_root}/temperature_processed'
    vector_file = f'{data_root}/riverflow_evaporation_604catchments_1970_2015.parquet'
    static_attr_file = f'{data_root}/Catchment_attributes/Catchment_attributes_nrfa.csv'
    land_mask_path = f'{data_root}/gb_temp_valid_mask_290x180.pt'

    # Static attributes to use
    static_attrs = [
        "latitude", "longitude",
        "minimum-altitude", "maximum-altitude", "50-percentile-altitude",
        "10-percentile-altitude", "90-percentile-altitude",
        "catchment-area", "dpsbar",
        "propwet", "bfihost",
    ]

    # Time periods
    train_start = '1970-01-01'  # ⭐ UPDATED: Changed from 1989 to 1970 (riverflow缺失1970-1988)
    train_end = '2010-12-31'
    val_start = '2011-01-01'
    val_end = '2015-12-30'  # Data ends on 2015-12-30, not 2015-12-31

    # Dataset settings
    stride = 20  # Stride for sliding window sampling (days)
    stats_cache_path = 'cache/normalization_stats.pt'

    # ========== Merged H5 Files (优化后) ==========
    # Training period h5 files
    # ⭐ UPDATED: Changed from 1989 to 1970 (需要准备1970-2010的merged h5文件)
    precip_train_h5 = f'{data_root}/precipitation_train_1970_2010.h5'
    soil_train_h5 = f'{data_root}/soil_moisture_train_1970_2010.h5'
    temp_train_h5 = f'{data_root}/temperature_train_1970_2010.h5'

    # Validation period h5 files
    precip_val_h5 = f'{data_root}/precipitation_val_2011_2015.h5'
    soil_val_h5 = f'{data_root}/soil_moisture_val_2011_2015.h5'
    temp_val_h5 = f'{data_root}/temperature_val_2011_2015.h5'

    # Performance optimization
    cache_images_to_memory = True  # 是否缓存图像到内存
    num_workers = 12  # 增加workers（原来是4）

    # ========== Mask Configuration (可调节的超参数) ==========
    image_mask_ratio = 0.4  # Ratio of patches to mask for image modalities
    vector_mask_ratio = 0.4  # Ratio of timesteps to mask for vector modalities
    land_threshold = 0.5  # Minimum land coverage for valid patches

    # ========== Training Hyperparameters ==========
    # Optimization
    batch_size = 16  # Batch size per GPU
    learning_rate = 1e-4  # Learning rate
    weight_decay = 0.05  # Weight decay
    betas = (0.9, 0.95)  # Adam betas
    eps = 1e-8  # Adam epsilon

    # Gradient clipping (CRITICAL for stability with FP16)
    gradient_clip_norm = 1.0  # Clip gradients to max norm of 1.0

    # Task loss weights (to balance different modalities)
    # Adjust these if some losses are much smaller/larger than others
    task_weights = {
        'precip_loss': 100.0,    # Precipitation tends to have very small values
        'soil_loss': 10.0,       # Soil moisture has moderate values
        'temp_loss': 10.0,       # Temperature has moderate values
        'evap_loss': 1.0,        # Evaporation baseline
        'riverflow_loss': 1.0,   # Riverflow baseline
    }

    # Training schedule
    epochs = 10  # Total number of epochs
    warmup_epochs = 5  # Number of warmup epochs
    min_lr = 0.0  # Minimum learning rate after decay

    # ========== Distributed Training ==========
    use_deepspeed = True  # Use DeepSpeed for distributed training
    zero_stage = 1  # DeepSpeed ZeRO optimization stage
    use_fp16 = True  # Use mixed precision training (FP16)
    gradient_accumulation_steps = 1  # Gradient accumulation steps
    # num_workers is now defined in Performance optimization section above

    # FP16 Loss Scaling Configuration (to prevent underflow/overflow)
    initial_scale_power = 16  # Initial loss scale = 2^16 = 65536
    loss_scale_window = 1000  # Consecutive steps before increasing loss scale
    min_loss_scale = 1.0      # Minimum loss scale
    hysteresis = 2            # Number of overflows before decreasing scale

    # ========== Logging and Checkpointing ==========
    # WandB settings
    use_wandb = True  # Use Weights & Biases for logging
    wandb_project = "water-mae-pretraining"  # WandB project name
    wandb_entity = None  # WandB entity (username or team), None = default

    # Checkpointing
    output_dir = "output/multimodal_mae"  # Directory to save checkpoints
    checkpoint_frequency = 10  # Save checkpoint every N epochs
    keep_last_n_checkpoints = 10  # Keep only last N checkpoints

    # Logging
    log_frequency = 10  # Log training stats every N batches
    val_frequency = 1  # Run validation every N epochs

    # ========== Validation ==========
    val_batch_size = 16  # Batch size for validation (can be larger)

    def __repr__(self):
        """Pretty print configuration"""
        lines = ["\n" + "=" * 60]
        lines.append("Multi-modal MAE Configuration")
        lines.append("=" * 60)

        sections = {
            "Model Architecture": [
                "d_model", "decoder_dim", "img_encoder_layers",
                "vec_encoder_layers", "decoder_layers", "nhead", "dropout"
            ],
            "Mask Configuration": [
                "image_mask_ratio", "vector_mask_ratio"
            ],
            "Training": [
                "batch_size", "learning_rate", "epochs", "warmup_epochs"
            ],
            "Distributed Training": [
                "use_deepspeed", "zero_stage", "use_fp16"
            ],
        }

        for section, keys in sections.items():
            lines.append(f"\n{section}:")
            for key in keys:
                value = getattr(self, key)
                lines.append(f"  {key}: {value}")

        lines.append("=" * 60 + "\n")
        return "\n".join(lines)


# Default configuration instance
default_config = MAEConfig()


if __name__ == '__main__':
    # Print configuration
    config = MAEConfig()
    print(config)

    # Print all attributes
    print("\nAll Configuration Parameters:")
    print("-" * 60)
    for key in sorted(dir(config)):
        if not key.startswith('_'):
            value = getattr(config, key)
            if not callable(value):
                print(f"{key}: {value}")
