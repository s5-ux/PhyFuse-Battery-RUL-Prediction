"""Hybrid model training pipeline for EV battery RUL prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from data_preprocessing import DataConfig, load_and_preprocess
from feature_engineering import EVConfig, add_ev_features, feature_columns


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    ev: EVConfig = field(default_factory=EVConfig)
    artifact_dir: str = "artifacts"
    random_state: int = 42
    mode: Literal["demo", "strict"] = "demo"


def artifact_name_for_mode(mode: str) -> str:
    suffix = "demo" if mode == "demo" else "strict"
    return f"battery_rul_hybrid_artifacts_{suffix}.joblib"


def _build_direct_model(random_state: int = 42):
    # XGBoost/LightGBM preferred. GradientBoosting is portable fallback.
    return GradientBoostingRegressor(
        n_estimators=720,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        min_samples_leaf=8,
        random_state=random_state,
    )


def _build_soh_model(random_state: int = 42):
    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )


def _build_quantile_model(alpha: float, random_state: int = 42):
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        min_samples_leaf=10,
        random_state=random_state,
    )


def _derive_rul_from_soh(
    soh_pct: np.ndarray,
    degradation_rate: np.ndarray,
    total_degradation: np.ndarray,
    eol_soh_pct: float = 80.0,
) -> np.ndarray:
    """Non-linear SOH-to-RUL conversion using exponential degradation law.

    SOH(t) = SOH0 * exp(-k * t)
    RUL = (1/k) * ln(SOH_current / SOH_EOL)
    """

    soh_frac = np.clip(soh_pct / 100.0, eol_soh_pct / 100.0 + 1e-6, 1.05)
    eol_frac = eol_soh_pct / 100.0

    # Blended stress term: local fade rate + cycle/calendar stress.
    k = 0.00008 + 0.12 * np.clip(degradation_rate, 1e-6, None) + 0.00025 * np.clip(total_degradation, 0.0, None)
    k = np.clip(k, 1e-5, 0.02)

    rul = (1.0 / k) * np.log(soh_frac / eol_frac)
    return np.clip(rul, 0.0, None)


def cycles_to_months(rul_cycles: np.ndarray, avg_daily_km: float = 135.0, km_per_cycle: float = 160.0) -> np.ndarray:
    km_remaining = rul_cycles * km_per_cycle
    days = km_remaining / max(avg_daily_km, 1e-6)
    return days / 30.0


def cycles_to_remaining_energy(rul_cycles: np.ndarray, pack_energy_kwh: float, soc_window: float = 0.8) -> np.ndarray:
    return rul_cycles * pack_energy_kwh * soc_window


def _risk_zone(rul_cycles: float) -> str:
    if rul_cycles > 1000:
        return "Green"
    if rul_cycles >= 500:
        return "Yellow"
    return "Red"


def _top_degradation_driver(model: GradientBoostingRegressor, cols: list[str]) -> str:
    """Identify dominant degradation driver family from feature importances."""

    imp = pd.Series(model.feature_importances_, index=cols)
    grouped = {
        "temperature": imp[[c for c in cols if "temp" in c or "thermal" in c or "calendar_stress" in c]].sum(),
        "DoD": imp[[c for c in cols if "dod" in c or "cycle_stress" in c]].sum(),
        "charging": imp[[c for c in cols if "c_rate" in c or "charge" in c or "fast_charge" in c]].sum(),
    }
    return max(grouped, key=grouped.get)


def _group_contribution(model: GradientBoostingRegressor, cols: list[str]) -> Dict[str, float]:
    imp = pd.Series(model.feature_importances_, index=cols)
    groups = {
        "temperature": float(imp[[c for c in cols if "temp" in c or "thermal" in c or "calendar_stress" in c]].sum()),
        "DoD": float(imp[[c for c in cols if "dod" in c or "cycle_stress" in c]].sum()),
        "charging": float(imp[[c for c in cols if "c_rate" in c or "charge" in c or "fast_charge" in c]].sum()),
    }
    total = sum(groups.values()) + 1e-9
    return {k: 100.0 * v / total for k, v in groups.items()}


def _groupkfold_cv_rmse(X: pd.DataFrame, y: pd.Series, groups: pd.Series, random_state: int) -> float:
    gkf = GroupKFold(n_splits=5)
    rmses = []
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        m = _build_direct_model(random_state)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = m.predict(X.iloc[te_idx])
        rmses.append(float(np.sqrt(mean_squared_error(y.iloc[te_idx], p))))
    return float(np.mean(rmses))


def _tail_sample_weights(y: pd.Series) -> np.ndarray:
    """Upweight target tails so the regressor learns low/high-RUL regimes better."""

    y_np = y.to_numpy(dtype=float)
    q10, q90 = np.quantile(y_np, [0.10, 0.90])
    center = float(np.median(y_np))
    spread = max(float(np.std(y_np)), 1e-6)

    # Smoothly increase weight away from the median, with extra tail emphasis.
    z = np.abs((y_np - center) / spread)
    w = 1.0 + 0.45 * z
    w = np.where((y_np <= q10) | (y_np >= q90), w * 1.8, w)
    return np.clip(w, 1.0, 4.0)


def _strict_group_split(df: pd.DataFrame, cfg: TrainConfig) -> dict[str, pd.DataFrame]:
    """Split by battery_id to avoid battery identity leakage across folds."""

    battery_ids = np.sort(df["battery_id"].unique())
    n = len(battery_ids)
    n_train = max(1, int(n * cfg.data.train_ratio))
    n_val = max(1, int(n * cfg.data.val_ratio))

    train_ids = set(battery_ids[:n_train])
    val_ids = set(battery_ids[n_train : n_train + n_val])
    test_ids = set(battery_ids[n_train + n_val :])
    if not test_ids:
        test_ids = set(battery_ids[-1:])

    train_df = df[df["battery_id"].isin(train_ids)].copy()
    val_df = df[df["battery_id"].isin(val_ids)].copy()
    test_df = df[df["battery_id"].isin(test_ids)].copy()
    return {"train": train_df, "val": val_df, "test": test_df}


def _metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    bias = float(np.mean(err))
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    ss_tot = float(np.sum(np.square(y_true - np.mean(y_true))))
    r2 = float(1.0 - np.sum(np.square(err)) / max(ss_tot, 1e-9))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "Slope": float(slope),
        "Intercept": float(intercept),
        "R2": r2,
    }


def train_pipeline(cfg: TrainConfig) -> Dict[str, object]:
    base_df, default_splits = load_and_preprocess(cfg.data)

    if cfg.mode == "strict":
        splits = _strict_group_split(base_df, cfg)
    else:
        splits = default_splits

    train_df = add_ev_features(splits["train"], cfg.ev)
    val_df = add_ev_features(splits["val"], cfg.ev)
    test_df = add_ev_features(splits["test"], cfg.ev)

    cols = feature_columns(mode=cfg.mode)

    X_train = train_df[cols]
    X_val = val_df[cols]
    X_test = test_df[cols]
    feature_medians = X_train.median(numeric_only=True).to_dict()

    y_train_rul = train_df["RUL_cycles"]
    y_train_soh = train_df["SOH_pct"]
    w_rul = _tail_sample_weights(y_train_rul)

    direct_model = _build_direct_model(cfg.random_state)
    direct_model.fit(X_train, y_train_rul, sample_weight=w_rul)

    # Baseline model for leaderboard-style comparison.
    baseline_cols = ["cycle_index", "calendar_age_days", "cumulative_energy_kwh"]
    baseline_model = LinearRegression()
    baseline_model.fit(X_train[baseline_cols], y_train_rul)

    soh_model = _build_soh_model(cfg.random_state)
    soh_model.fit(X_train, y_train_soh)

    q_lower_model = _build_quantile_model(alpha=0.10, random_state=cfg.random_state)
    q_upper_model = _build_quantile_model(alpha=0.90, random_state=cfg.random_state)
    q_lower_model.fit(X_train, y_train_rul, sample_weight=w_rul)
    q_upper_model.fit(X_train, y_train_rul, sample_weight=w_rul)

    # Learn a simple calibration map on validation to reduce conservative bias.
    val_pred_direct = direct_model.predict(X_val)
    val_pred_soh = np.clip(soh_model.predict(X_val), 0.0, 100.0)
    val_pred_hybrid = _derive_rul_from_soh(
        soh_pct=val_pred_soh,
        degradation_rate=val_df["degradation_rate"].values,
        total_degradation=val_df["total_degradation"].values,
        eol_soh_pct=cfg.ev.eol_soh_pct,
    )
    val_raw_blend = 0.85 * val_pred_direct + 0.15 * val_pred_hybrid
    cal_model = LinearRegression()
    cal_model.fit(val_raw_blend.reshape(-1, 1), val_df["RUL_cycles"].values)

    test_out = test_df.copy()
    max_rul_cap = float(train_df["RUL_cycles"].max() * 1.25)

    test_out["RUL_pred_direct"] = direct_model.predict(X_test)
    test_out["SOH_pred_pct"] = np.clip(soh_model.predict(X_test), 0.0, 100.0)

    test_out["RUL_pred_hybrid"] = _derive_rul_from_soh(
        soh_pct=test_out["SOH_pred_pct"].values,
        degradation_rate=test_out["degradation_rate"].values,
        total_degradation=test_out["total_degradation"].values,
        eol_soh_pct=cfg.ev.eol_soh_pct,
    )
    test_out["RUL_pred_hybrid"] = np.clip(test_out["RUL_pred_hybrid"], 0.0, None)

    # Weighted blend reduces conservative shrink from simple averaging.
    raw_mean = 0.85 * test_out["RUL_pred_direct"] + 0.15 * test_out["RUL_pred_hybrid"]
    test_out["RUL_mean"] = np.clip(cal_model.predict(raw_mean.values.reshape(-1, 1)), 0.0, max_rul_cap)
    test_out["RUL_lower_bound"] = np.clip(q_lower_model.predict(X_test), 0.0, max_rul_cap)
    test_out["RUL_upper_bound"] = np.clip(q_upper_model.predict(X_test), 0.0, max_rul_cap)

    spread = (test_out["RUL_upper_bound"] - test_out["RUL_lower_bound"]).clip(lower=0.0)
    test_out["confidence_score"] = (1.0 - spread / test_out["RUL_mean"].clip(lower=1.0)).clip(0.0, 1.0)

    test_out["RUL_months"] = cycles_to_months(test_out["RUL_mean"].values, avg_daily_km=cfg.ev.avg_daily_km_default)
    test_out["RUL_energy_kwh"] = cycles_to_remaining_energy(
        test_out["RUL_mean"].values,
        pack_energy_kwh=cfg.ev.pack_energy_kwh,
        soc_window=cfg.ev.soc_max - cfg.ev.soc_min,
    )

    test_out["risk_zone"] = test_out["RUL_mean"].apply(_risk_zone)
    test_out["warranty_risk"] = np.where(test_out["RUL_mean"] < 500, "High", np.where(test_out["RUL_mean"] < 1000, "Medium", "Low"))
    test_out["maintenance_recommendation"] = test_out["RUL_months"].apply(lambda m: f"Replace battery in {max(m, 0.0):.1f} months")

    abnormal_threshold = float(train_df["degradation_rate"].quantile(0.95) * 1.25)
    test_out["anomaly_flag"] = np.where(test_out["degradation_rate"] > abnormal_threshold, "Abnormal degradation", "Normal")
    test_out["failing_within_3_months"] = test_out["RUL_months"] <= 3.0

    top_driver = _top_degradation_driver(direct_model, cols)
    driver_contrib = _group_contribution(direct_model, cols)
    explainability_summary = f"High {top_driver} contributes {driver_contrib[top_driver]:.1f}% of degradation signal"
    feature_importance = pd.DataFrame(
        {
            "feature": cols,
            "importance": direct_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # Cross-battery generalization score using group-based split.
    cv_df = add_ev_features(base_df.copy(), cfg.ev)
    X_cv = cv_df[cols]
    y_cv = cv_df["RUL_cycles"]
    g_cv = cv_df["battery_id"]
    group_cv_rmse = _groupkfold_cv_rmse(X_cv, y_cv, g_cv, cfg.random_state)

    baseline_pred = baseline_model.predict(X_test[baseline_cols])
    baseline_rmse = float(np.sqrt(mean_squared_error(test_out["RUL_cycles"], baseline_pred)))
    direct_rmse = float(np.sqrt(mean_squared_error(test_out["RUL_cycles"], test_out["RUL_pred_direct"])))
    performance_summary = _metric_summary(
        test_out["RUL_cycles"].to_numpy(dtype=float),
        test_out["RUL_mean"].to_numpy(dtype=float),
    )
    leaderboard = pd.DataFrame(
        [
            {"model": "Linear baseline", "RMSE": baseline_rmse},
            {"model": "Hybrid direct model", "RMSE": direct_rmse},
        ]
    )

    artifacts = {
        "direct_model": direct_model,
        "soh_model": soh_model,
        "q_lower_model": q_lower_model,
        "q_upper_model": q_upper_model,
        "baseline_model": baseline_model,
        "calibration_model": cal_model,
        "feature_columns": cols,
        "feature_medians": feature_medians,
        "blend_weights": {"direct": 0.85, "hybrid": 0.15},
        "target_transform": "none",
        "mode": cfg.mode,
        "data_config": asdict(cfg.data),
        "ev_config": asdict(cfg.ev),
        "abnormal_threshold": abnormal_threshold,
        "top_degradation_driver": top_driver,
        "driver_contribution": driver_contrib,
        "explainability_summary": explainability_summary,
        "group_cv_rmse": group_cv_rmse,
        "performance_summary": performance_summary,
        "leaderboard": leaderboard,
        "feature_importance": feature_importance,
        "test_predictions": test_out,
    }

    out_dir = Path(cfg.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = artifact_name_for_mode(cfg.mode)
    joblib.dump(artifacts, out_dir / artifact_name)
    test_out.to_csv(out_dir / f"test_predictions_{cfg.mode}.csv", index=False)

    return artifacts


def predict_single(features: pd.DataFrame, artifacts: Dict[str, object]) -> pd.DataFrame:
    cols = artifacts["feature_columns"]
    ev_cfg = artifacts["ev_config"]

    x = features.reindex(columns=cols).copy()
    x = x.apply(pd.to_numeric, errors="coerce")

    fill_values = artifacts.get("feature_medians")
    if not isinstance(fill_values, dict) or not fill_values:
        # Backward-compatible fallback for older artifacts.
        ref_df = artifacts.get("test_predictions")
        if isinstance(ref_df, pd.DataFrame):
            fill_values = ref_df.reindex(columns=cols).median(numeric_only=True).to_dict()
        else:
            fill_values = {}

    x = x.fillna(fill_values)
    x = x.fillna(0.0)

    blend_weights = artifacts.get("blend_weights", {"direct": 0.5, "hybrid": 0.5})
    w_direct = float(blend_weights.get("direct", 0.5))
    w_hybrid = float(blend_weights.get("hybrid", 0.5))
    w_sum = max(w_direct + w_hybrid, 1e-6)
    w_direct /= w_sum
    w_hybrid /= w_sum

    max_rul_cap = float(artifacts.get("test_predictions")["RUL_cycles"].max() * 1.25)
    target_transform = artifacts.get("target_transform", "none")
    direct_raw = artifacts["direct_model"].predict(x)
    if target_transform == "log1p":
        pred_direct = np.expm1(direct_raw)
    else:
        pred_direct = direct_raw
    pred_soh = np.clip(artifacts["soh_model"].predict(x), 0.0, 100.0)

    degradation_rate = np.clip(x["degradation_rate"].values, 1e-6, None)
    total_degradation = np.clip(x["total_degradation"].values, 0.0, None)
    pred_hybrid = _derive_rul_from_soh(
        pred_soh,
        degradation_rate,
        total_degradation,
        eol_soh_pct=ev_cfg["eol_soh_pct"],
    )

    pred_hybrid = np.clip(pred_hybrid, 0.0, None)
    raw_pred = (w_direct * pred_direct) + (w_hybrid * pred_hybrid)
    cal_model = artifacts.get("calibration_model")
    if cal_model is not None:
        mean_pred = np.clip(cal_model.predict(raw_pred.reshape(-1, 1)), 0.0, max_rul_cap)
    else:
        mean_pred = np.clip(raw_pred, 0.0, max_rul_cap)
    lower_raw = artifacts["q_lower_model"].predict(x)
    upper_raw = artifacts["q_upper_model"].predict(x)
    if target_transform == "log1p":
        lower = np.clip(np.expm1(lower_raw), 0.0, max_rul_cap)
        upper = np.clip(np.expm1(upper_raw), 0.0, max_rul_cap)
    else:
        lower = np.clip(lower_raw, 0.0, max_rul_cap)
        upper = np.clip(upper_raw, 0.0, max_rul_cap)
    confidence = np.clip(1.0 - (upper - lower) / np.clip(mean_pred, 1.0, None), 0.0, 1.0)

    out = pd.DataFrame(
        {
            "RUL_mean": mean_pred,
            "RUL_lower_bound": lower,
            "RUL_upper_bound": upper,
            "RUL_months": cycles_to_months(mean_pred),
            "RUL_energy_kwh": cycles_to_remaining_energy(mean_pred, pack_energy_kwh=ev_cfg["pack_energy_kwh"], soc_window=ev_cfg["soc_max"] - ev_cfg["soc_min"]),
            "confidence_score": confidence,
            "anomaly_flag": np.where(degradation_rate > float(artifacts.get("abnormal_threshold", 1e-3)), "Abnormal degradation", "Normal"),
            "degradation_phase": np.where(np.clip(pred_soh / 100.0, 0.0, 1.0) > 0.90, "early", np.where(np.clip(pred_soh / 100.0, 0.0, 1.0) > 0.85, "mid", "late")),
        }
    )
    return out


def train_all_modes(base_cfg: TrainConfig | None = None) -> dict[str, Dict[str, object]]:
    cfg = base_cfg or TrainConfig()
    out: dict[str, Dict[str, object]] = {}
    for mode in ["demo", "strict"]:
        mode_cfg = TrainConfig(
            data=cfg.data,
            ev=cfg.ev,
            artifact_dir=cfg.artifact_dir,
            random_state=cfg.random_state,
            mode=mode,
        )
        out[mode] = train_pipeline(mode_cfg)
    return out


def load_artifacts_for_mode(mode: str, artifact_dir: str = "artifacts") -> Dict[str, object]:
    p = Path(artifact_dir) / artifact_name_for_mode(mode)
    return joblib.load(p)


def comparison_table_from_artifacts(artifacts_by_mode: dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for mode, artifacts in artifacts_by_mode.items():
        summary = artifacts.get("performance_summary", {})
        rows.append(
            {
                "mode": mode,
                "R2": float(summary.get("R2", 0.0)),
                "MAE": float(summary.get("MAE", 0.0)),
                "RMSE": float(summary.get("RMSE", 0.0)),
                "Bias": float(summary.get("Bias", 0.0)),
                "Slope": float(summary.get("Slope", 0.0)),
                "GroupCV_RMSE": float(artifacts.get("group_cv_rmse", 0.0)),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_artifacts = train_all_modes(TrainConfig())
    for mode, artifacts in all_artifacts.items():
        print(f"Saved artifacts to artifacts/{artifact_name_for_mode(mode)}")
        print(f"Generated predictions for {len(artifacts['test_predictions'])} test cycles ({mode})")
