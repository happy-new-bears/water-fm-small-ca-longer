"""
Example Custom Configuration for Experiment 2
- Smaller learning rate (more conservative)
- Standard batch size
- Lower mask ratio (easier task)
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
    train_start = '1989-01-01'
    train_end = '2010-12-31'
    val_start = '2011-01-01'
    val_end = '2015-12-30'

    # Dataset settings
    stride = 20  # Stride for sliding window sampling (days)
    stats_cache_path = 'cache/normalization_stats.pt'

    # ========== Merged H5 Files ==========
    # Training period h5 files
    precip_train_h5 = f'{data_root}/precipitation_train_1989_2010.h5'
    soil_train_h5 = f'{data_root}/soil_moisture_train_1989_2010.h5'
    temp_train_h5 = f'{data_root}/temperature_train_1989_2010.h5'

    # Validation period h5 files
    precip_val_h5 = f'{data_root}/precipitation_val_2011_2015.h5'
    soil_val_h5 = f'{data_root}/soil_moisture_val_2011_2015.h5'
    temp_val_h5 = f'{data_root}/temperature_val_2011_2015.h5'

    # Performance optimization
    cache_images_to_memory = True
    num_workers = 12

    # ========== EXPERIMENT 2: Conservative approach ==========
    # CHANGED: Lower mask ratio (easier task)
    image_mask_ratio = 0.3  # Lower mask ratio (vs 0.4 in default)
    vector_mask_ratio = 0.3  # Lower mask ratio (vs 0.4 in default)
    land_threshold = 0.5

    # ========== Training Hyperparameters ==========
    # Standard batch size
    batch_size = 16  # Same as default

    # CHANGED: Lower learning rate (more conservative)
    learning_rate = 5e-5  # Lower learning rate (vs 1e-4 in default)

    betas = (0.9, 0.95)
    eps = 1e-8
    weight_decay = 0.05

    # Training settings
    epochs = 10
    val_batch_size = 8
    val_frequency = 1  # Validate every N epochs
    checkpoint_frequency = 10  # Save checkpoint every N epochs
    keep_last_n_checkpoints = 2  # Keep only last N checkpoints
    log_frequency = 10  # Log every N batches

    # ========== DeepSpeed Configuration ==========
    zero_stage = 1  # ZeRO stage (0, 1, 2, or 3)
    use_fp16 = True  # Use mixed precision training
    gradient_accumulation_steps = 1

    # ========== Output Configuration ==========
    output_dir = 'outputs'  # Base output directory (will create timestamped subdirs)

    # ========== WandB Configuration ==========
    use_wandb = True
    wandb_project = 'multimodal-mae'
    wandb_entity = None  # Your wandb username/team
