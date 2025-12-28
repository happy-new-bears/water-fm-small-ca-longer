# Changelog: Support for Partially Missing Riverflow Data (1970-1988)

## Problem

Training data period extended from 1989-2010 to 1970-2010, but:
- **Riverflow**: Missing 1970-1988, available 1989-2010
- **Evaporation**: Complete 1970-2010
- **Image modalities**: Complete 1970-2010

Original code would filter out all catchments because riverflow had 100% NaN ratio in 1970-1988.

## Solution

Implemented per-variable validity range checking during data loading.

---

## Modified Files

### 1. `datasets/data_utils.py`

#### `interpolate_features()` function
**Added parameter**: `variable_validity_ranges: dict = None`

**New behavior**:
- If `variable_validity_ranges` is provided, check each variable's NaN ratio **only within its valid range**
- Example: `{'discharge_vol': (datetime(1989,1,1), datetime(2010,12,31))}`
- Riverflow NaN ratio checked only in 1989-2010 (ignores 1970-1988 completely)
- Evaporation NaN ratio checked in full 1970-2010 range
- **Intersection logic**: Only keep catchments that are valid for ALL variables

**Old behavior** (if `None`): All variables checked over entire (start, end) range

#### `load_vector_data_from_parquet()` function
**Added parameter**: `variable_validity_ranges: dict = None`

**New behavior**:
- Accepts per-variable validity ranges
- Builds complete validity dict (defaults to full range for unspecified variables)
- Prints validity ranges for transparency
- Passes to `interpolate_features()`

---

### 2. `train_mae.py`

#### `create_datasets()` function

**Added logic**:
```python
# Define validity ranges
riverflow_valid_start = datetime(1989, 1, 1)
variable_validity_ranges = {
    'evaporation': (train_start, val_end),      # 1970-2015
    'discharge_vol': (riverflow_valid_start, val_end),  # 1989-2015
}

# Pass to load function
vector_data, time_vec, catchment_ids, var_names = load_vector_data_from_parquet(
    ...,
    variable_validity_ranges=variable_validity_ranges
)
```

---

## Data Flow

### Example: 604 catchments, 1970-2010 training period

**Step 1: Filter catchments by NaN ratio in valid ranges**
```
Checking evaporation in 1970-01-01 to 2015-12-30:
  - Catchment 001: 2.3% NaN → Valid
  - Catchment 002: 8.7% NaN → Invalid (> 5% threshold)
  ...
  - Result: 550/604 catchments valid

Checking discharge_vol in 1989-01-01 to 2015-12-30:  # ⭐ Note: Starts from 1989!
  - Catchment 001: 1.5% NaN → Valid
  - Catchment 003: 12.4% NaN → Invalid
  ...
  - Result: 520/604 catchments valid

Intersection: 500 catchments valid for BOTH variables
```

**Step 2: Load data for valid catchments**
```
Data loaded: [num_days=16802, num_catchments=500, num_vars=2]
  - 1970-1988: evaporation has data, riverflow is NaN
  - 1989-2010: both evaporation and riverflow have data
```

**Step 3: Dataset marks samples with invalid riverflow**
```python
# In multimodal_dataset_optimized.py __getitem__
start_date = self.date_list[start_day_idx]
riverflow_valid = (start_date >= datetime(1989, 1, 1))

# Sample from 1975-03-15: riverflow_valid = False
# Sample from 1995-06-20: riverflow_valid = True
```

---

## Backward Compatibility

✅ **Fully backward compatible**

If `variable_validity_ranges=None` (default):
- All variables checked over entire (start, end) range
- Identical behavior to original code

---

## Testing

### Test Case 1: Verify catchment filtering
```python
# Should find catchments valid for evap in 1970-2010 AND riverflow in 1989-2010
# Not zero catchments (old bug)
```

### Test Case 2: Verify data shape
```python
vector_data.shape  # Should be [num_days_1970_2015, num_catchments, 2]
# 1970-1988: riverflow column is NaN
# 1989-2015: both columns have data
```

### Test Case 3: Verify sample validity flag
```python
# Sample from 1975: riverflow_valid = False
# Sample from 1995: riverflow_valid = True
```

---

## Related Changes

This works in conjunction with:

1. **Dataset layer** (`multimodal_dataset_optimized.py`):
   - Added `riverflow_valid` flag based on sample start date

2. **Collate layer** (`collate.py`):
   - Collects `riverflow_valid_mask` for batch

3. **Model layer** (`vector_encoder.py`, `multimodal_mae.py`):
   - Uses `valid_mask` to set padding_mask=True for invalid samples
   - Prevents gradient flow from other modalities to riverflow encoder

4. **Decoder layers** (`image_decoder.py`, `vector_decoder.py`, `layers.py`):
   - Use global `encoder_padding_mask` in cross-attention
   - Invalid riverflow tokens don't contribute to reconstruction

---

## Summary

**What was changed**:
- Added per-variable validity range checking in data loading
- Riverflow NaN ratio checked only in 1989-2015 (not 1970-1988)
- Evaporation NaN ratio checked in full 1970-2015

**Why**:
- Enable training on extended 1970-2010 period
- Properly handle partially missing modality (riverflow 1970-1988)
- Maintain catchment ID consistency across time periods

**Impact**:
- ✅ More training data (41 years vs 21 years for image/evap)
- ✅ Riverflow encoder trained on 21 years (1989-2010)
- ✅ Other modalities benefit from 41 years (1970-2010)
- ✅ Backward compatible (default behavior unchanged)
