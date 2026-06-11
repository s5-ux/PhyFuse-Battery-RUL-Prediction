"""Kalman-based SOC estimation evaluation for EV battery analytics.

Implements:
1) Standard Kalman Filter (linearized measurement model)
2) Extended Kalman Filter (nonlinear OCV-SOC model)
3) Adaptive EKF (innovation-driven Q/R adaptation)

Also quantifies downstream RUL impact using filtered SOC features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import DataConfig, create_capacity_target, load_raw_data, standardize_columns


@dataclass
class BatteryFilterConfig:
    capacity_ah: float = 2.8
    dt_s: float = 60.0
    eta_coulombic: float = 0.995
    q_soc: float = 1e-5
    q_r0: float = 5e-7
    r_voltage: float = 2.5e-3
    r0_init: float = 0.045
    soc_init: float = 0.92


def ocv_model(soc: float, temp_c: float, aging: float) -> float:
    """Nonlinear OCV model: polynomial + thermal + aging effect."""

    s = float(np.clip(soc, 1e-6, 0.999999))
    base = 3.0 + 0.85 * s - 0.16 * (s**2) + 0.08 * np.log(s / (1.0 - s))
    thermal = 0.0015 * (temp_c - 25.0)
    aging_drop = 0.025 * aging
    return base + thermal - aging_drop


def d_ocv_dsoc(soc: float) -> float:
    s = float(np.clip(soc, 1e-6, 0.999999))
    return 0.85 - 0.32 * s + 0.08 * (1.0 / s + 1.0 / (1.0 - s))


class StandardKalmanFilter:
    """Linear KF with state x=[SOC, R0].

    State model:
    x_k = A x_{k-1} + B u_k + w_k

    Measurement model (linearized):
    z_k = H_k x_k + c_k + v_k
    where z is terminal voltage.
    """

    def __init__(self, cfg: BatteryFilterConfig):
        self.cfg = cfg
        self.x = np.array([cfg.soc_init, cfg.r0_init], dtype=float)
        self.P = np.diag([2e-2, 8e-4]).astype(float)
        self.Q = np.diag([cfg.q_soc, cfg.q_r0]).astype(float)
        self.R = np.array([[cfg.r_voltage]], dtype=float)
        self.h_slope = d_ocv_dsoc(0.5)
        self.soc_lin_ref = 0.5

    def predict(self, current_a: float):
        a = np.eye(2)
        b = np.array([[-self.cfg.eta_coulombic * self.cfg.dt_s / (3600.0 * self.cfg.capacity_ah)], [0.0]])
        u = np.array([[current_a]], dtype=float)
        self.x = a @ self.x + (b @ u).reshape(-1)
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        self.x[1] = float(np.clip(self.x[1], 0.005, 0.25))
        self.P = a @ self.P @ a.T + self.Q

    def update(self, voltage_v: float, current_a: float, temp_c: float, aging: float) -> Tuple[np.ndarray, np.ndarray]:
        h = np.array([[self.h_slope, -current_a]], dtype=float)
        c = ocv_model(self.soc_lin_ref, temp_c, aging) - self.h_slope * self.soc_lin_ref

        z_pred = float((h @ self.x)[0] + c)
        y = np.array([[voltage_v - z_pred]], dtype=float)
        s = h @ self.P @ h.T + self.R
        k = self.P @ h.T @ np.linalg.inv(s)

        self.x = self.x + (k @ y).reshape(-1)
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        self.x[1] = float(np.clip(self.x[1], 0.005, 0.25))

        i = np.eye(2)
        self.P = (i - k @ h) @ self.P
        return y.reshape(-1), k[:, 0]


class ExtendedKalmanFilter:
    """EKF with nonlinear voltage measurement equation.

    h(x,u,T,aging) = OCV(SOC,T,aging) - I * R0
    """

    def __init__(self, cfg: BatteryFilterConfig):
        self.cfg = cfg
        self.x = np.array([cfg.soc_init, cfg.r0_init], dtype=float)
        self.P = np.diag([2e-2, 8e-4]).astype(float)
        self.Q = np.diag([cfg.q_soc, cfg.q_r0]).astype(float)
        self.R = np.array([[cfg.r_voltage]], dtype=float)

    def predict(self, current_a: float):
        soc_next = self.x[0] - self.cfg.eta_coulombic * current_a * self.cfg.dt_s / (3600.0 * self.cfg.capacity_ah)
        self.x[0] = float(np.clip(soc_next, 0.0, 1.0))
        self.x[1] = float(np.clip(self.x[1], 0.005, 0.25))
        f = np.eye(2)
        self.P = f @ self.P @ f.T + self.Q

    def update(self, voltage_v: float, current_a: float, temp_c: float, aging: float) -> Tuple[np.ndarray, np.ndarray]:
        soc = float(np.clip(self.x[0], 1e-6, 0.999999))
        r0 = float(self.x[1])

        z_pred = ocv_model(soc, temp_c, aging) - current_a * r0
        h = np.array([[d_ocv_dsoc(soc), -current_a]], dtype=float)

        y = np.array([[voltage_v - z_pred]], dtype=float)
        s = h @ self.P @ h.T + self.R
        k = self.P @ h.T @ np.linalg.inv(s)

        self.x = self.x + (k @ y).reshape(-1)
        self.x[0] = float(np.clip(self.x[0], 0.0, 1.0))
        self.x[1] = float(np.clip(self.x[1], 0.005, 0.25))

        i = np.eye(2)
        self.P = (i - k @ h) @ self.P
        return y.reshape(-1), k[:, 0]


class AdaptiveExtendedKalmanFilter(ExtendedKalmanFilter):
    """AEKF with innovation-based adaptive Q and R.

    R_k <- (1-beta)R_{k-1} + beta * innov^2
    Q_k <- (1-alpha)Q_{k-1} + alpha * (K * innov)(K * innov)^T
    """

    def __init__(self, cfg: BatteryFilterConfig, alpha: float = 0.02, beta: float = 0.03):
        super().__init__(cfg)
        self.alpha = alpha
        self.beta = beta

    def update(self, voltage_v: float, current_a: float, temp_c: float, aging: float) -> Tuple[np.ndarray, np.ndarray]:
        y, k = super().update(voltage_v, current_a, temp_c, aging)

        innov = float(y[0])
        self.R = (1.0 - self.beta) * self.R + self.beta * np.array([[innov**2 + 1e-8]])

        dk = k.reshape(-1, 1)
        q_add = (dk @ dk.T) * (innov**2)
        self.Q = (1.0 - self.alpha) * self.Q + self.alpha * q_add

        # Keep adaptation bounded for numerical stability in production BMS use.
        self.Q[0, 0] = float(np.clip(self.Q[0, 0], 1e-8, 5e-3))
        self.Q[1, 1] = float(np.clip(self.Q[1, 1], 1e-9, 1e-3))
        self.R[0, 0] = float(np.clip(self.R[0, 0], 1e-7, 5e-2))
        return y, k


def _load_base_df(cfg: DataConfig) -> pd.DataFrame:
    raw = load_raw_data(cfg.data_path)
    base = standardize_columns(raw)
    base = create_capacity_target(base, eol_capacity_pct=cfg.eol_capacity_pct)
    base = base.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)
    return base


def build_soc_sequence(base: pd.DataFrame, cfg: BatteryFilterConfig) -> pd.DataFrame:
    seq = base.copy()
    discharge_h = np.clip(seq["discharge_time_s"] / 3600.0, 1e-5, None)
    seq["current_a"] = np.clip(cfg.capacity_ah / discharge_h, 0.05, 6.0)

    seq["temp_c"] = np.clip(25.0 + 0.012 * seq["cycle_index"], 20.0, 48.0)
    seq["aging"] = np.clip((seq["cycle_index"] / seq["cycle_index"].max()), 0.0, 1.0)

    # Build a realistic SOC trajectory per battery with repeated partial recharge behavior.
    soc_vals = np.zeros(len(seq), dtype=float)
    for _, idx in seq.groupby("battery_id").groups.items():
        ids = np.array(list(idx), dtype=int)
        soc = 0.95
        for j, rid in enumerate(ids):
            i_a = float(seq.iloc[rid]["current_a"])
            aging = float(seq.iloc[rid]["aging"])
            usable_capacity = cfg.capacity_ah * (1.0 - 0.20 * aging)
            usable_capacity = max(usable_capacity, 0.6 * cfg.capacity_ah)
            dsoc = cfg.eta_coulombic * i_a * cfg.dt_s / (3600.0 * usable_capacity)
            soc = soc - dsoc
            if soc < 0.10:
                soc = 0.95 - 0.08 * np.sin(j / 6.0)
            soc_vals[rid] = float(np.clip(soc, 0.02, 0.99))

    seq["soc_true"] = soc_vals

    rng = np.random.default_rng(42)
    v_nom = [
        ocv_model(s, t, a) - i * (0.04 + 0.02 * a)
        for s, t, a, i in zip(seq["soc_true"], seq["temp_c"], seq["aging"], seq["current_a"])
    ]
    seq["voltage_v"] = np.array(v_nom, dtype=float) + rng.normal(0.0, 0.008, size=len(seq))
    return seq


def inject_edge_cases(seq: pd.DataFrame) -> pd.DataFrame:
    out = seq.copy()
    rng = np.random.default_rng(7)

    spike_idx = rng.choice(len(out), size=max(10, len(out) // 80), replace=False)
    out.loc[spike_idx, "current_a"] *= rng.uniform(1.8, 2.8, size=len(spike_idx))

    miss_idx = rng.choice(len(out), size=max(15, len(out) // 70), replace=False)
    out.loc[miss_idx, "voltage_v"] = np.nan

    drift_start = int(0.55 * len(out))
    drift_end = int(0.75 * len(out))
    out.loc[drift_start:drift_end, "temp_c"] += 8.0

    age_start = int(0.70 * len(out))
    out.loc[age_start:, "aging"] = np.clip(out.loc[age_start:, "aging"] + 0.18, 0.0, 1.0)
    return out


def run_filter_on_sequence(seq: pd.DataFrame, flt) -> Dict[str, np.ndarray]:
    n = len(seq)
    soc_est = np.zeros(n, dtype=float)
    r0_est = np.zeros(n, dtype=float)
    innov = np.zeros(n, dtype=float)
    k_soc = np.zeros(n, dtype=float)

    for i, row in enumerate(seq.itertuples(index=False)):
        current_a = float(getattr(row, "current_a"))
        temp_c = float(getattr(row, "temp_c"))
        aging = float(getattr(row, "aging"))
        voltage = float(getattr(row, "voltage_v")) if pd.notna(getattr(row, "voltage_v")) else np.nan

        flt.predict(current_a)
        if np.isfinite(voltage):
            y, k = flt.update(voltage, current_a, temp_c, aging)
            innov[i] = float(y[0])
            k_soc[i] = float(k[0])
        else:
            innov[i] = np.nan
            k_soc[i] = np.nan

        soc_est[i] = float(flt.x[0])
        r0_est[i] = float(flt.x[1])

    return {
        "soc_est": soc_est,
        "r0_est": r0_est,
        "innovation": innov,
        "k_soc": k_soc,
    }


def soc_metrics(true_soc: np.ndarray, est_soc: np.ndarray) -> Dict[str, float]:
    err = est_soc - true_soc
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))

    abs_err = np.abs(err)
    threshold = 0.02
    converged = np.where(abs_err <= threshold)[0]
    conv_idx = int(converged[0]) if len(converged) else len(err) - 1

    stability = float(np.nanstd(err[-max(200, len(err) // 5) :]))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "ConvergenceStep": float(conv_idx),
        "StabilityStd": stability,
    }


def _split_by_group(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gids = np.sort(df["battery_id"].unique())
    split = max(1, int(len(gids) * 0.70))
    train_ids = set(gids[:split])
    test_ids = set(gids[split:]) if split < len(gids) else set(gids[-1:])
    tr = df[df["battery_id"].isin(train_ids)].copy()
    te = df[df["battery_id"].isin(test_ids)].copy()
    tr = tr.dropna(subset=[target_col])
    te = te.dropna(subset=[target_col])
    return tr, te


def evaluate_rul_with_filtered_soc(seq: pd.DataFrame, results: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    df = seq.copy()
    df["soc_kf"] = results["KF"]["soc_est"]
    df["soc_ekf"] = results["EKF"]["soc_est"]
    df["soc_aekf"] = results["AEKF"]["soc_est"]

    target = "RUL_cycles"
    base_feats = ["cycle_index", "voltage_v", "current_a", "temp_c", "aging"]

    scores = []
    feature_sets = {
        "raw": base_feats,
        "raw_plus_kf": base_feats + ["soc_kf"],
        "raw_plus_ekf": base_feats + ["soc_ekf"],
        "raw_plus_aekf": base_feats + ["soc_aekf"],
    }

    train_df, test_df = _split_by_group(df, target)

    for name, cols in feature_sets.items():
        x_train = train_df[cols].copy()
        x_test = test_df[cols].copy()
        med = x_train.median(numeric_only=True)
        x_train = x_train.fillna(med).fillna(0.0)
        x_test = x_test.fillna(med).fillna(0.0)

        y_train = train_df[target].to_numpy(dtype=float)
        y_test = test_df[target].to_numpy(dtype=float)

        m = GradientBoostingRegressor(n_estimators=350, learning_rate=0.04, max_depth=4, random_state=42)
        m.fit(x_train, y_train)
        p = m.predict(x_test)

        scores.append(
            {
                "feature_set": name,
                "MAE": float(mean_absolute_error(y_test, p)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, p))),
                "R2": float(r2_score(y_test, p)),
            }
        )

    return pd.DataFrame(scores)


def plot_filter_diagnostics(seq: pd.DataFrame, results: Dict[str, Dict[str, np.ndarray]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(seq))
    soc_true = seq["soc_true"].to_numpy(dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    axes[0].plot(t, soc_true, color="black", linewidth=1.6, label="True SOC")
    axes[0].plot(t, results["KF"]["soc_est"], alpha=0.85, label="KF")
    axes[0].plot(t, results["EKF"]["soc_est"], alpha=0.85, label="EKF")
    axes[0].plot(t, results["AEKF"]["soc_est"], alpha=0.9, linewidth=1.4, label="Adaptive EKF")
    axes[0].set_ylabel("SOC")
    axes[0].set_title("True SOC vs KF/EKF/Adaptive EKF estimates")
    axes[0].grid(True, linestyle=":", alpha=0.35)
    axes[0].legend(loc="best")

    for name, color in [("KF", "#457b9d"), ("EKF", "#e76f51"), ("AEKF", "#2a9d8f")]:
        err = results[name]["soc_est"] - soc_true
        axes[1].plot(t, err, color=color, linewidth=1.0, label=f"{name} error")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("SOC error")
    axes[1].set_title("Estimation error over time")
    axes[1].grid(True, linestyle=":", alpha=0.35)
    axes[1].legend(loc="best")

    for name, color in [("KF", "#457b9d"), ("EKF", "#e76f51"), ("AEKF", "#2a9d8f")]:
        axes[2].plot(t, results[name]["k_soc"], color=color, linewidth=1.0, label=f"{name} K_soc")
    axes[2].set_ylabel("Kalman gain")
    axes[2].set_title("Kalman gain evolution (SOC channel)")
    axes[2].grid(True, linestyle=":", alpha=0.35)
    axes[2].legend(loc="best")

    for name, color in [("KF", "#457b9d"), ("EKF", "#e76f51"), ("AEKF", "#2a9d8f")]:
        axes[3].plot(t, results[name]["innovation"], color=color, linewidth=1.0, alpha=0.85, label=f"{name} innovation")
    axes[3].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[3].set_xlabel("Time step")
    axes[3].set_ylabel("Voltage residual")
    axes[3].set_title("Innovation (measurement residual)")
    axes[3].grid(True, linestyle=":", alpha=0.35)
    axes[3].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "soc_filter_diagnostics.png", dpi=170)
    plt.close(fig)


def plot_comparison_tables(soc_table: pd.DataFrame, rul_table: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(soc_table))
    ax.bar(x - 0.18, soc_table["RMSE"], width=0.35, label="SOC RMSE", color="#457b9d")
    ax.bar(x + 0.18, soc_table["MAE"], width=0.35, label="SOC MAE", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(soc_table["filter"])
    ax.set_ylabel("SOC error")
    ax.set_title("SOC estimation accuracy across filters")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "soc_filter_metrics.png", dpi=170)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 4.6))
    xr = np.arange(len(rul_table))
    ax2.bar(xr - 0.18, rul_table["RMSE"], width=0.35, label="RUL RMSE", color="#2a9d8f")
    ax2.bar(xr + 0.18, rul_table["MAE"], width=0.35, label="RUL MAE", color="#8d99ae")
    ax2.set_xticks(xr)
    ax2.set_xticklabels(rul_table["feature_set"], rotation=15)
    ax2.set_ylabel("RUL error (cycles)")
    ax2.set_title("Downstream RUL impact: raw vs filtered SOC features")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(out_dir / "rul_with_filtered_soc_metrics.png", dpi=170)
    plt.close(fig2)


def run_kalman_evaluation(data_cfg: DataConfig | None = None, filt_cfg: BatteryFilterConfig | None = None) -> Dict[str, pd.DataFrame]:
    data_cfg = data_cfg or DataConfig()
    filt_cfg = filt_cfg or BatteryFilterConfig()

    base = _load_base_df(data_cfg)
    seq = build_soc_sequence(base, filt_cfg)
    seq = inject_edge_cases(seq)

    kf = StandardKalmanFilter(filt_cfg)
    ekf = ExtendedKalmanFilter(filt_cfg)
    aekf = AdaptiveExtendedKalmanFilter(filt_cfg)

    results = {
        "KF": run_filter_on_sequence(seq, kf),
        "EKF": run_filter_on_sequence(seq, ekf),
        "AEKF": run_filter_on_sequence(seq, aekf),
    }

    soc_true = seq["soc_true"].to_numpy(dtype=float)
    soc_rows = []
    for name in ["KF", "EKF", "AEKF"]:
        m = soc_metrics(soc_true, results[name]["soc_est"])
        m["filter"] = name
        soc_rows.append(m)
    soc_table = pd.DataFrame(soc_rows)[["filter", "MAE", "RMSE", "ConvergenceStep", "StabilityStd"]]

    rul_table = evaluate_rul_with_filtered_soc(seq, results)

    out_dir = Path("artifacts") / "kalman_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_filter_diagnostics(seq, results, out_dir)
    plot_comparison_tables(soc_table, rul_table, out_dir)

    soc_table.to_csv(out_dir / "soc_filter_metrics.csv", index=False)
    rul_table.to_csv(out_dir / "rul_filtered_feature_comparison.csv", index=False)

    return {
        "soc_table": soc_table,
        "rul_table": rul_table,
    }


if __name__ == "__main__":
    outputs = run_kalman_evaluation()
    print("SOC filter comparison:")
    print(outputs["soc_table"].to_string(index=False))
    print("\nRUL impact with filtered SOC:")
    print(outputs["rul_table"].to_string(index=False))
    print("\nSaved plots and tables under artifacts/kalman_eval")
