"""Feature engineering for EV pack degradation modeling under Indian conditions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EVConfig:
    """Domain constants for EV pack behavior and Indian operating conditions."""

    pack_energy_kwh: float = 35.0
    nominal_pack_voltage_v: float = 350.0
    nominal_cell_capacity_ah: float = 2.8
    eol_soh_pct: float = 80.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    avg_daily_km_default: float = 135.0
    avg_speed_kmph_default: float = 28.0


def _rolling(df: pd.DataFrame, col: str, win: int, stat: str) -> pd.Series:
    if stat == "mean":
        return df[col].rolling(win, min_periods=1).mean()
    return df[col].rolling(win, min_periods=1).std().fillna(0.0)


def _midc_cycle_energy_kwh() -> float:
    """Simulate an MIDC-like 1080-second urban speed trace and return cycle energy."""

    t = np.arange(0, 1080)
    speed_kmph = np.zeros_like(t, dtype=float)

    # Piecewise speed profile with repeated stop-go pulses.
    speed_kmph[:180] = np.maximum(0.0, 22.0 * np.sin(np.linspace(0, np.pi, 180)))
    speed_kmph[180:360] = np.maximum(0.0, 30.0 * np.sin(np.linspace(0, 1.3 * np.pi, 180)))
    speed_kmph[360:540] = np.maximum(0.0, 26.0 * np.sin(np.linspace(0, np.pi, 180)))
    speed_kmph[540:720] = np.maximum(0.0, 34.0 * np.sin(np.linspace(0, 1.2 * np.pi, 180)))
    speed_kmph[720:900] = np.maximum(0.0, 24.0 * np.sin(np.linspace(0, np.pi, 180)))
    speed_kmph[900:1080] = np.maximum(0.0, 28.0 * np.sin(np.linspace(0, 1.1 * np.pi, 180)))

    speed_mps = speed_kmph / 3.6
    acc_mps2 = np.gradient(speed_mps)
    acc_pos = np.clip(acc_mps2, 0.0, None)

    # Power proxy: rolling resistance + acceleration burden.
    power_kw = (0.16 * speed_mps + 40.0 * acc_pos * speed_mps) / 1000.0
    energy_kwh = np.trapz(power_kw, dx=1.0) / 3600.0
    return float(max(energy_kwh, 0.01))


def midc_speed_profile() -> pd.DataFrame:
    """Return MIDC-like speed-time and power overlay for dashboard visualization."""

    t = np.arange(0, 1080)
    speed_kmph = np.zeros_like(t, dtype=float)
    speed_kmph[:180] = np.maximum(0.0, 22.0 * np.sin(np.linspace(0, np.pi, 180)))
    speed_kmph[180:360] = np.maximum(0.0, 30.0 * np.sin(np.linspace(0, 1.3 * np.pi, 180)))
    speed_kmph[360:540] = np.maximum(0.0, 26.0 * np.sin(np.linspace(0, np.pi, 180)))
    speed_kmph[540:720] = np.maximum(0.0, 34.0 * np.sin(np.linspace(0, 1.2 * np.pi, 180)))
    speed_kmph[720:900] = np.maximum(0.0, 24.0 * np.sin(np.linspace(0, np.pi, 180)))
    speed_kmph[900:1080] = np.maximum(0.0, 28.0 * np.sin(np.linspace(0, 1.1 * np.pi, 180)))

    speed_mps = speed_kmph / 3.6
    acc_mps2 = np.gradient(speed_mps)
    acc_pos = np.clip(acc_mps2, 0.0, None)
    power_kw = (0.16 * speed_mps + 40.0 * acc_pos * speed_mps) / 1000.0

    return pd.DataFrame({"time_s": t, "speed_kmph": speed_kmph, "power_kw": power_kw})


def add_ev_features(df: pd.DataFrame, cfg: EVConfig, rolling_window_n: int = 20) -> pd.DataFrame:
    """Create mandatory battery, thermal, usage, charging, driving and aging features.

    Many signals are estimated proxies because the source dataset is lab-level cycle data.
    """

    out = df.copy()

    if "battery_id" not in out.columns:
        out["battery_id"] = 0

    out = out.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)

    # Battery and BMS features
    out["cell_voltage_mean"] = (out["max_voltage_discharge_v"] + out["min_voltage_charge_v"]) / 2.0
    out["cell_voltage_std"] = (
        (out["max_voltage_discharge_v"] - out["min_voltage_charge_v"]).abs() / 6.0
    ).clip(0.001, None)
    out["pack_voltage"] = out["cell_voltage_mean"] * (cfg.nominal_pack_voltage_v / 3.6)

    # Approximate current from discharge time and nominal cell capacity.
    discharge_hours = (out["discharge_time_s"] / 3600.0).clip(lower=1e-5)
    out["avg_current"] = (cfg.nominal_cell_capacity_ah / discharge_hours).clip(0.1, 5.0 * cfg.nominal_cell_capacity_ah)
    out["peak_current"] = out["avg_current"] * 1.35
    out["regen_current"] = out["avg_current"] * 0.20

    # Internal resistance proxy from dynamic voltage span over current.
    dv = (out["max_voltage_discharge_v"] - out["min_voltage_charge_v"]).abs().clip(lower=1e-4)
    out["internal_resistance"] = (dv / out["avg_current"]).clip(0.001, 0.5)

    # Thermal features under Indian range 20-45C using cycle behavior proxy.
    norm_cycle = out["cycle_index"] / max(float(out["cycle_index"].max()), 1.0)
    charge_dev = (out["charging_time_s"] - out["charging_time_s"].median()) / (out["charging_time_s"].std() + 1e-6)
    out["avg_temp"] = (25.0 + 12.0 * norm_cycle + 2.5 * charge_dev).clip(20.0, 45.0)
    out["max_temp"] = (out["avg_temp"] + 2.0 + 0.5 * np.abs(charge_dev)).clip(21.0, 50.0)
    out["temp_gradient"] = (out["max_temp"] - out["avg_temp"]).clip(0.0, None)
    out["time_above_40C"] = ((out["avg_temp"] - 40.0).clip(lower=0.0) / 5.0) * out["discharge_time_s"]

    # Usage features
    out["dod"] = (out["discharge_time_s"] / out["discharge_time_s"].quantile(0.95)).clip(0.05, 1.0)
    out["soc_end"] = cfg.soc_min
    out["soc_start"] = (out["soc_end"] + out["dod"]).clip(cfg.soc_min, cfg.soc_max)
    out["avg_soc"] = (out["soc_start"] + out["soc_end"]) / 2.0
    out["soc_variance"] = ((out["soc_start"] - out["avg_soc"]) ** 2 + (out["soc_end"] - out["avg_soc"]) ** 2) / 2.0
    out["dod_rolling_avg"] = out.groupby("battery_id")["dod"].transform(lambda s: s.rolling(rolling_window_n, min_periods=1).mean())
    out["dod_variance"] = out.groupby("battery_id")["dod"].transform(lambda s: s.rolling(rolling_window_n, min_periods=1).var().fillna(0.0))

    # Charging features
    pack_capacity_ah = cfg.pack_energy_kwh * 1000.0 / cfg.nominal_pack_voltage_v
    out["c_rate"] = (out["avg_current"] / cfg.nominal_cell_capacity_ah).clip(0.2, 3.0)
    out["fast_charge_ratio"] = (out["c_rate"] > 1.0).astype(float)
    out["avg_charging_time"] = out.groupby("battery_id")["charging_time_s"].transform(lambda s: s.rolling(rolling_window_n, min_periods=1).mean())

    # Driving behavior proxies for MIDC-like stop-go urban profile.
    out["avg_speed"] = (cfg.avg_speed_kmph_default - 3.0 * out["dod"] + 0.05 * (45.0 - out["avg_temp"])).clip(12.0, 45.0)
    out["stop_go_ratio"] = (0.35 + 0.4 * (1.0 - out["avg_speed"] / 45.0)).clip(0.2, 0.9)
    out["acceleration_events"] = (out["stop_go_ratio"] * 80 + out["dod"] * 25).round()
    out["regen_braking_events"] = (out["acceleration_events"] * 0.65).round()

    # MIDC integration: cycle energy and stress score from an explicit 1080s profile.
    midc_base_energy = _midc_cycle_energy_kwh()
    out["midc_energy_per_cycle"] = midc_base_energy * (
        1.0 + 0.45 * out["stop_go_ratio"] + 0.002 * out["acceleration_events"]
    )
    out["midc_stress_score"] = out["midc_energy_per_cycle"] * (
        1.0 + 0.03 * (out["avg_temp"] - 25.0).clip(lower=0.0)
    )

    # Aging features
    out["calendar_age_days"] = out["cycle_index"]
    out["idle_time_ratio"] = (1.0 - out["discharge_time_s"] / (out["discharge_time_s"] + out["charging_time_s"] + 1e-6)).clip(0.0, 1.0)
    out["energy_throughput_kwh"] = out["dod"] * cfg.pack_energy_kwh * (cfg.soc_max - cfg.soc_min)
    out["cumulative_energy_kwh"] = out.groupby("battery_id")["energy_throughput_kwh"].cumsum()

    # Advanced rolling statistics
    base_roll_cols = ["dod", "avg_temp", "c_rate", "internal_resistance", "capacity"]
    for col in base_roll_cols:
        out[f"{col}_rolling_mean_10"] = out.groupby("battery_id")[col].transform(lambda s: s.rolling(10, min_periods=1).mean())
        out[f"{col}_rolling_mean_50"] = out.groupby("battery_id")[col].transform(lambda s: s.rolling(50, min_periods=1).mean())
        out[f"{col}_rolling_std_10"] = out.groupby("battery_id")[col].transform(lambda s: s.rolling(10, min_periods=1).std().fillna(0.0))

    # Degradation indicators
    initial_capacity = out.groupby("battery_id")["capacity"].transform(lambda s: s.iloc[: min(len(s), 10)].mean())
    out["capacity_fade"] = initial_capacity - out["capacity"]
    delta_capacity = out.groupby("battery_id")["capacity"].diff().fillna(0.0)
    delta_cycles = out.groupby("battery_id")["cycle_index"].diff().replace(0, np.nan).fillna(1.0)
    out["degradation_rate"] = (-delta_capacity / delta_cycles).clip(lower=1e-6)

    # Physics-inspired feature and thermal stress index
    out["degradation_score"] = np.exp(out["avg_temp"] / 40.0) * np.power(out["dod"], 1.3) * np.power(out["c_rate"], 1.1)
    out["thermal_stress"] = out["time_above_40C"] * out["avg_temp"]

    # Explicit cycle-aging and calendar-aging separation for hybrid physics modeling.
    out["cycle_stress"] = np.power(out["dod"], 1.3) * np.power(out["c_rate"], 1.1)
    out["calendar_stress"] = np.exp((out["avg_temp"] - 25.0) / 10.0) * (out["calendar_age_days"] / 365.0)
    out["total_degradation"] = out["cycle_stress"] + 0.3 * out["calendar_stress"]

    # Ensure RUL naming consistency for downstream models.
    out["RUL_cycles"] = out["RUL_cycles"].clip(lower=0)
    out["SOH_pct"] = out["capacity"] * 100.0
    out["SOH_frac"] = out["SOH_pct"].clip(0.0, 100.0) / 100.0

    out["degradation_phase"] = np.where(
        out["SOH_frac"] > 0.90,
        "early",
        np.where(out["SOH_frac"] > 0.85, "mid", "late"),
    )
    out["degradation_phase_code"] = out["degradation_phase"].map({"early": 0, "mid": 1, "late": 2}).astype(int)

    # Edge-case indicators help the model separate extreme degradation regimes.
    out["high_temp_flag"] = (out["avg_temp"] >= 40.0).astype(int)
    out["high_dod_flag"] = (out["dod"] >= 0.80).astype(int)
    out["low_soh_flag"] = (out["SOH_frac"] <= 0.85).astype(int)

    return out


def feature_columns(mode: str = "demo") -> list[str]:
    """Model input feature list."""

    cols = [
        "battery_id",
        "cycle_index",
        "cell_voltage_mean",
        "cell_voltage_std",
        "pack_voltage",
        "avg_current",
        "peak_current",
        "regen_current",
        "internal_resistance",
        "avg_temp",
        "max_temp",
        "temp_gradient",
        "time_above_40C",
        "soc_start",
        "soc_end",
        "avg_soc",
        "soc_variance",
        "dod",
        "dod_rolling_avg",
        "dod_variance",
        "c_rate",
        "fast_charge_ratio",
        "avg_charging_time",
        "avg_speed",
        "stop_go_ratio",
        "acceleration_events",
        "regen_braking_events",
        "midc_energy_per_cycle",
        "midc_stress_score",
        "calendar_age_days",
        "idle_time_ratio",
        "cumulative_energy_kwh",
        "capacity_fade",
        "degradation_rate",
        "degradation_score",
        "thermal_stress",
        "cycle_stress",
        "calendar_stress",
        "total_degradation",
        "degradation_phase_code",
        "high_temp_flag",
        "high_dod_flag",
        "low_soh_flag",
        "dod_rolling_mean_10",
        "dod_rolling_mean_50",
        "dod_rolling_std_10",
        "avg_temp_rolling_mean_10",
        "avg_temp_rolling_mean_50",
        "avg_temp_rolling_std_10",
        "c_rate_rolling_mean_10",
        "c_rate_rolling_mean_50",
        "c_rate_rolling_std_10",
        "internal_resistance_rolling_mean_10",
        "internal_resistance_rolling_mean_50",
        "internal_resistance_rolling_std_10",
        "capacity_rolling_mean_10",
        "capacity_rolling_mean_50",
        "capacity_rolling_std_10",
    ]

    if mode == "strict":
        # Strict mode excludes features directly tied to identity/target-derived SOH proxies.
        drop_cols = {
            "battery_id",
            "capacity_fade",
            "capacity_rolling_mean_10",
            "capacity_rolling_mean_50",
            "capacity_rolling_std_10",
            "degradation_phase_code",
            "low_soh_flag",
        }
        return [c for c in cols if c not in drop_cols]

    return cols
