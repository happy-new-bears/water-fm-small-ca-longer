# Water Foundation Model - Multi-modal MAE for Hydrological Prediction

A multi-modal Masked Autoencoder (MAE) for hydrological time series prediction, supporting image modalities (precipitation, soil moisture, temperature) and vector modalities (evaporation, river flow) with static catchment attributes.

## Key Features

### Architecture
- **CrossMAE-style decoder**: Efficient cross-attention instead of full self-attention
- **Multi-modal fusion**: Shared transformer layers for cross-modal learning
- **FiLM conditioning**: Static catchment attributes modulate vector encoders
- **Patch-based encoding**:
  - Images: 10×10 spatial patches
  - Vectors: 8-catchment spatial patches with temporal tokens

### Data Handling
- **Training period**: 1970-2010 (41 years)
  - Image modalities: Complete 1970-2010
  - Evaporation: Complete 1970-2010
  - River flow: **1989-2010 only** (1970-1988 missing)
- **Validation period**: 2011-2015
- **Automatic handling of missing riverflow data**:
  - Samples from 1970-1988: riverflow encoder runs but tokens are masked via padding_mask
  - No gradient flow from other modalities to riverflow encoder for invalid samples
  - Loss only computed on valid riverflow samples (1989-2010)

### Performance Optimizations
- Pre-normalized data (computed once, cached)
- Merged HDF5 files for fast image loading
- Vectorized operations (no batch loops)
- Optional memory caching
- Multi-worker data loading

## Project Structure

```
.
├── models/
│   ├── multimodal_mae.py       # Main model with shared fusion
│   ├── image_encoder.py        # ViT-style image encoder
│   ├── vector_encoder.py       # Vector encoder with FiLM
│   ├── image_decoder.py        # CrossMAE decoder for images
│   ├── vector_decoder.py       # CrossMAE decoder for vectors
│   └── layers.py               # Shared layers (CrossAttention, FiLM, etc.)
├── datasets/
│   ├── multimodal_dataset_optimized.py  # Optimized dataset
│   └── collate.py              # MAE-style masking collate function
├── configs/
│   └── mae_config.py           # Configuration
├── train_mae.py                # Training script
└── merge_h5.py                 # Data preprocessing script

```

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/water_fm_small_ca_longer.git
cd water_fm_small_ca_longer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install numpy pandas h5py pyarrow
pip install wandb  # For logging (optional)
pip install deepspeed  # For distributed training (optional)
```

## Data Preparation

### Required Data Files

1. **Image modalities** (1970-2010):
   - `precipitation_train_1970_2010.h5`
   - `soil_moisture_train_1970_2010.h5`
   - `temperature_train_1970_2010.h5`

2. **Vector modalities** (1970-2015):
   - `riverflow_evaporation_604catchments_1970_2015.parquet`
     - Evaporation: Complete data 1970-2015
     - River flow: Data available 1989-2015 (missing 1970-1988)

3. **Static attributes**:
   - `Catchment_attributes_nrfa.csv`

4. **Land mask**:
   - `gb_temp_valid_mask_290x180.pt`

### Merge Images into HDF5

```bash
python merge_h5.py --start 1970-01-01 --end 2010-12-31 --modality precip
python merge_h5.py --start 1970-01-01 --end 2010-12-31 --modality soil
python merge_h5.py --start 1970-01-01 --end 2010-12-31 --modality temp
```

## Training

### Basic Training

```bash
python train_mae.py
```

### With DeepSpeed (Distributed Training)

```bash
deepspeed --num_gpus=4 train_mae.py
```

### Key Configuration Parameters

Edit `configs/mae_config.py`:

```python
# Data paths
data_root = '/path/to/your/data'

# Training period (riverflow valid from 1989-01-01)
train_start = '1970-01-01'
train_end = '2010-12-31'

# Model architecture
d_model = 256
decoder_dim = 128
img_encoder_layers = 6
vec_encoder_layers = 4

# Masking
image_mask_ratio = 0.4
vector_mask_ratio = 0.4

# Training
batch_size = 16
learning_rate = 1e-4
epochs = 100
```

## Implementation Details

### Handling Missing Riverflow Data (1970-1988)

The model uses a sophisticated masking strategy to handle partially missing modalities:

1. **Dataset Layer**:
   - Marks samples with `riverflow_valid = False` if start date < 1989-01-01

2. **Encoder Layer**:
   - Invalid samples still go through riverflow encoder (maintains batch shape)
   - All tokens set to `padding_mask = True` for invalid samples

3. **Shared Fusion**:
   - Invalid riverflow tokens included in `fused_features` but masked
   - Self-attention ignores these tokens (attention weight = 0)

4. **Decoder Layer**:
   - Cross-attention uses global `encoder_padding_mask`
   - Invalid tokens don't contribute to any modality's reconstruction

5. **Loss Computation**:
   - Riverflow loss multiplied by `valid_sample_mask`
   - Invalid samples contribute 0 to riverflow loss

This approach ensures:
- ✅ Batch dimension consistency (no ragged tensors)
- ✅ Gradient isolation (no backprop to riverflow encoder from other modalities)
- ✅ Riverflow encoder trained only on valid data (1989-2010)
- ✅ Other modalities benefit from full 1970-2010 dataset

### CrossMAE Efficiency

Standard MAE: All tokens (visible + masked) do self-attention → O(N²)
CrossMAE: Only masked tokens as queries, attend to visible → O(M×N)

Speedup: ~3-4x when mask_ratio=0.75

## Model Architecture

```
Input:
├─ Images [B, T, 290, 180]
│   └─ Patchify → [B, T, 522, 100]
└─ Vectors [B, 76, 8, T]

Encoders (separate):
├─ Image Encoders (3×)
│   ├─ Patch projection
│   ├─ Position embedding (spatial + temporal)
│   ├─ Transformer layers (6)
│   └─ Remove masked patches
└─ Vector Encoders (2×)
    ├─ Linear projection
    ├─ Position embedding (spatial + temporal)
    ├─ FiLM-modulated Transformer (4 layers)
    └─ Remove masked tokens

Shared Fusion:
├─ Concatenate all visible tokens
├─ Shared Transformer layers (2)
└─ → fused_features [B, L_total, 256]

Decoders (separate):
├─ Image Decoders (3×)
│   ├─ Create masked queries
│   ├─ Cross-attention to fused_features
│   └─ Predict masked patches
└─ Vector Decoders (2×)
    ├─ Create masked queries
    ├─ Cross-attention to fused_features
    └─ Predict masked values

Output:
├─ Reconstructed images
└─ Reconstructed vectors
```

## Loss Function

```python
total_loss = (
    task_weights['precip_loss'] * precip_loss +
    task_weights['soil_loss'] * soil_loss +
    task_weights['temp_loss'] * temp_loss +
    task_weights['evap_loss'] * evap_loss +
    task_weights['riverflow_loss'] * riverflow_loss * valid_sample_mask
)
```

Default task weights:
- Precipitation: 100.0 (very small values)
- Soil: 10.0
- Temperature: 10.0
- Evaporation: 1.0
- River flow: 1.0

## Citation

If you use this code, please cite:

```bibtex
@article{yourpaper2025,
  title={Multi-modal Masked Autoencoder for Hydrological Prediction},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- CrossMAE architecture inspired by [CrossMAE](https://github.com/TonyLianLong/CrossMAE)
- FiLM conditioning from [FiLM](https://arxiv.org/abs/1709.07871)
- MAE framework from [Masked Autoencoders](https://arxiv.org/abs/2111.06377)
