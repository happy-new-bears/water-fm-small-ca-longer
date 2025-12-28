"""
Utility functions for loading and processing data
"""

import numpy as np
import polars as pl
from datetime import datetime
from typing import Tuple, List
import logging


def interpolate_features(
    df: pl.DataFrame,
    start: datetime,
    end: datetime,
    nan_ratio: float,
    log_level: int = logging.INFO,
) -> pl.DataFrame:
    """
    Interpolate data in specified time range and nan ratio.

    Steps:
    1. Filter catchments with a NaN ratio upper bound
    2. Interpolate missing values for selected catchments during time period

    Args:
        df: DataFrame to be processed
        start: Start time of the time period (inclusive)
        end: End time of the time period (exclusive)
        nan_ratio: Maximum nan ratio allowed for selecting catchment
        log_level: Logging level

    Returns:
        Preprocessed DataFrame
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Filter catchments with NaN ratio below threshold
    time_expr = pl.col("date").ge(start) & pl.col("date").lt(end)
    valid_cid = set(
        cid[0]
        for cid, g in df.filter(time_expr).group_by("ID")
        if (g.null_count() < g.shape[0] * nan_ratio).to_numpy().all()
    )

    valid_df = df.filter(pl.col("ID").is_in(valid_cid))
    print(f"Found {valid_df['ID'].unique().len()} valid catchments after filtering NaN ratio")

    # Interpolate missing values
    interpolate_df = (
        valid_df.group_by("ID")
        .map_groups(
            lambda g: g.interpolate()
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")  # ensure no NaN left at start/end
        )
        .filter(time_expr)
    )
    print(f"Interpolated for {interpolate_df['ID'].unique().len()} catchments")

    return interpolate_df.with_columns(pl.col("ID").cast(pl.Int32))


def load_vector_data_from_parquet(
    fpath: str,
    variables: List[str],
    start: datetime,
    end: datetime,
    nan_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load vector data from parquet file and reshape to array format.

    Adapted from fm_v0/datasets.py get_vectors() function.

    Args:
        fpath: Path to parquet file
        variables: List of variable names (e.g., ['evaporation', 'riverflow'])
        start: Start date (datetime object)
        end: End date (datetime object)
        nan_ratio: Maximum allowed NaN ratio for interpolation (default: 0.05)

    Returns:
        data: np.ndarray of shape [num_days, num_catchments, num_vars]
        time_vec: np.ndarray of dates [num_days]
        catchment_ids: np.ndarray of catchment IDs [num_catchments]
        var_names: List of variable names in order
    """
    required_cols = ['date', 'ID']
    df_cols_to_load = list(set(required_cols + variables))

    # Read parquet file
    df = pl.read_parquet(fpath)

    try:
        df = df.select(df_cols_to_load)
    except pl.exceptions.ColumnNotFoundError as e:
        raise ValueError(
            f"Cols {df_cols_to_load} not in {fpath}. Error: {e}. "
            f"Available: {df.columns}"
        )

    # Handle NaN/null values
    df = df.fill_nan(None)

    # Ensure date column is datetime type
    if not isinstance(df["date"].dtype, pl.Datetime):
        df = df.with_columns(pl.col("date").cast(pl.Datetime))

    # Filter by date range
    df_filtered_by_date = df.filter(
        (pl.col("date") >= start.replace(hour=0, minute=0, second=0, microsecond=0))
        & (pl.col("date") < end.replace(hour=0, minute=0, second=0, microsecond=0))
    )

    if df_filtered_by_date.height == 0:
        raise ValueError(f"No vector data for date range {start} to {end} in {fpath}.")

    # Interpolate missing values
    df_processed = interpolate_features(
        df_filtered_by_date, start, end, nan_ratio
    ).sort(["date", "ID"])

    # Get variable names in order (excluding date and ID)
    vnames_ordered = [v for v in variables if v not in ["date", "ID"]]
    df_final_selection = df_processed.select(["date", "ID"] + vnames_ordered)

    # Get dimensions
    nid = df_final_selection["ID"].n_unique()
    nday = df_final_selection["date"].n_unique()

    if nid == 0 or nday == 0:
        raise ValueError("No unique catchments or days after processing vector data.")

    print(f"Loaded vector data: {nday} days, {nid} catchments. Variables: {vnames_ordered}.")

    # Convert to numpy and reshape
    npdata = df_final_selection.drop(["date", "ID"]).to_numpy()

    time_vec = (
        df_final_selection["date"].unique().sort().to_numpy().astype("datetime64[D]")
    )

    catchment_ids = df_final_selection["ID"].unique().sort().to_numpy()

    # Reshape: from [nday*nid, num_vars] to [nday, nid, num_vars]
    npdata_reshaped = npdata.reshape(nday, nid, len(vnames_ordered)).astype("float32")

    return npdata_reshaped, time_vec, catchment_ids, vnames_ordered
