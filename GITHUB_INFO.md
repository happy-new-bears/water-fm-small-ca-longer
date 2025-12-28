# GitHub Repository Information

## Repository Details

- **URL**: https://github.com/happy-new-bears/water-fm-small-ca-longer
- **Clone URL (SSH)**: `git@github.com-happybears:happy-new-bears/water-fm-small-ca-longer.git`
- **Clone URL (HTTPS)**: `https://github.com/happy-new-bears/water-fm-small-ca-longer.git`

## Initial Commit

```
commit 41a8aaf
Initial commit: Multi-modal MAE for Hydrological Prediction

Major features:
- Multi-modal architecture (3 image + 2 vector modalities)
- CrossMAE-style efficient decoder with cross-attention
- FiLM conditioning for static catchment attributes
- Support for partially missing modalities (riverflow 1989-2010)
- Optimized data loading with pre-normalization and HDF5 caching
- Vectorized operations throughout (no batch loops)

Training data period: 1970-2010 (riverflow valid from 1989)
Validation period: 2011-2015
```

## Files Included

Total: 24 files

### Core Code
- `models/` - Model architecture (7 files)
- `datasets/` - Data loading and preprocessing (7 files)
- `configs/` - Configuration files (4 files)
- `utils/` - Utility scripts (4 files)
- `train_mae.py` - Main training script

### Documentation
- `README.md` - Comprehensive documentation
- `.gitignore` - Git ignore rules

## Future Updates

### To push future changes:

```bash
# Stage changes
git add .

# Commit
git commit -m "Your commit message"

# Push
git push origin main
```

### To create a new branch:

```bash
# Create and switch to new branch
git checkout -b feature/your-feature-name

# Push to GitHub
git push -u origin feature/your-feature-name
```

### To pull latest changes:

```bash
git pull origin main
```

## Repository Settings Recommendations

### 1. Add Topics/Tags on GitHub
Visit: https://github.com/happy-new-bears/water-fm-small-ca-longer

Suggested tags:
- `deep-learning`
- `pytorch`
- `masked-autoencoder`
- `hydrology`
- `multi-modal`
- `time-series`
- `transformer`
- `climate-science`

### 2. Add Description
"Multi-modal Masked Autoencoder for hydrological prediction with support for partially missing modalities (1970-2010)"

### 3. (Optional) Add License
If you want to make it open source:
- MIT License (most permissive)
- Apache 2.0 (includes patent grant)
- GPL v3 (copyleft)

### 4. (Optional) Enable Features
- Issues: For bug tracking
- Discussions: For Q&A
- Projects: For roadmap/planning

## Important Notes

1. **Large files are ignored** (.h5, .parquet, .pt, .pth)
   - These should be stored separately (e.g., on cloud storage)
   - Add download instructions to README if needed

2. **SSH Configuration**
   - This repo uses SSH key: `~/.ssh/id_ed25519_happybears`
   - SSH host: `github.com-happybears`

3. **Cache files are ignored**
   - `cache/normalization_stats.pt` is not tracked
   - Users need to regenerate on first run

## Collaboration

To invite collaborators:
1. Go to: https://github.com/happy-new-bears/water-fm-small-ca-longer/settings/access
2. Click "Add people"
3. Enter their GitHub username
