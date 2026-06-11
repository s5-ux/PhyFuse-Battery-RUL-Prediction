"""Data loading and preprocessing for EV battery RUL modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    """Configuration for data handling and train/val/test splits."""

    data_path: str = "Battery_RUL.csv"
    eol_capacity_pct: float = 80.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV data."""

    df = pd.read_csv(path)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source column names and keep required base fields."""

    out = df.copy()
    out = out.rename(
        columns={
            "Cycle_Index": "cycle_index",
            "Discharge Time (s)": "discharge_time_s",
            "Decrement 3.6-3.4V (s)": "decrement_3p6_3p4_s",
            "Max. Voltage Dischar. (V)": "max_voltage_discharge_v",
            "Min. Voltage Charg. (V)": "min_voltage_charge_v",
            "Time at 4.15V (s)": "time_at_4p15v_s",
            "Time constant current (s)": "time_constant_current_s",
            "Charging time (s)": "charging_time_s",
            "RUL": "RUL_cycles",
        }
    )
    # Infer battery boundaries from cycle resets in raw order.
    cycle_reset = out["cycle_index"].diff().fillna(0) < 0
    out["battery_id"] = cycle_reset.cumsum().astype(int)

    # Fallback if cycle resets are not present in source order.
    if out["battery_id"].nunique() <= 1:
        n_groups = min(14, max(5, len(out) // 500))
        out["battery_id"] = pd.qcut(np.arange(len(out)), q=n_groups, labels=False, duplicates="drop").astype(int)

    out = out.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)

    numeric_cols = [
        "battery_id",
        "cycle_index",
        "discharge_time_s",
        "decrement_3p6_3p4_s",
        "max_voltage_discharge_v",
        "min_voltage_charge_v",
        "time_at_4p15v_s",
        "time_constant_current_s",
        "charging_time_s",
        "RUL_cycles",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=numeric_cols).reset_index(drop=True)
    return out


def clip_extremes(df: pd.DataFrame, cols: list[str], low: float = 0.01, high: float = 0.99) -> pd.DataFrame:
    """Clip heavy outliers to stabilize feature engineering and model training."""

    out = df.copy()
    for col in cols:
        q_low = out[col].quantile(low)
        q_high = out[col].quantile(high)
        out[col] = out[col].clip(q_low, q_high)
    return out


def create_capacity_target(df: pd.DataFrame, eol_capacity_pct: float = 80.0) -> pd.DataFrame:
    """Derive capacity/SOH from RUL so hybrid SOH->RUL logic can be learned.

    We approximate normalized SOH to match known end-of-life behavior:
    SOH = EOL + (100 - EOL) * (RUL / RUL_max)
    """

    out = df.copy()
    rul_max = max(float(out["RUL_cycles"].max()), 1.0)
    out["soh_pct"] = eol_capacity_pct + (100.0 - eol_capacity_pct) * (out["RUL_cycles"] / rul_max)
    out["capacity"] = out["soh_pct"] / 100.0
    return out


def split_time_ordered(df: pd.DataFrame, cfg: DataConfig) -> Dict[str, pd.DataFrame]:
    """Create time-ordered train/val/test splits."""

    n = len(df)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    return {"train": train_df, "val": val_df, "test": test_df}


def load_and_preprocess(cfg: DataConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """End-to-end preprocessing entrypoint."""

    raw = load_raw_data(cfg.data_path)
    base = standardize_columns(raw)
    base = clip_extremes(
        base,
        cols=[
            "discharge_time_s",
            "charging_time_s",
            "time_constant_current_s",
            "time_at_4p15v_s",
            "decrement_3p6_3p4_s",
        ],
    )
    base = create_capacity_target(base, eol_capacity_pct=cfg.eol_capacity_pct)
    splits = split_time_ordered(base, cfg)
    return base, splits
