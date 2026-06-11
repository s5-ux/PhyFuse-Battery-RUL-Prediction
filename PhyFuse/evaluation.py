"""Evaluation metrics and visualization utilities for EV battery RUL models."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    """Return key metrics for direct and hybrid predictions."""

    direct = regression_metrics(test_df["RUL_cycles"].values, test_df["RUL_pred_direct"].values)
    hybrid = regression_metrics(test_df["RUL_cycles"].values, test_df["RUL_pred_hybrid"].values)
    out = pd.DataFrame(
        [
            {"model": "Direct RUL", **direct},
            {"model": "Hybrid SOH->RUL", **hybrid},
        ]
    )
    return out


def plot_capacity_vs_cycles(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.plot(df["cycle_index"], df["capacity"], label="Capacity")
    plt.axhline(0.80, linestyle="--", color="red", label="EOL 80%")
    plt.xlabel("Cycle Index")
    plt.ylabel("Capacity (fraction)")
    plt.title("Capacity vs Cycles")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_soh_vs_time(df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 4.5))
    plt.plot(df["calendar_age_days"], df["SOH_pct"], color="#0077b6")
    plt.axhline(80.0, linestyle="--", color="red", label="EOL 80%")
    plt.xlabel("Calendar Age (days)")
    plt.ylabel("SOH (%)")
    plt.title("SOH vs Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rul_pred_vs_actual(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(df["RUL_cycles"], df["RUL_pred_direct"], alpha=0.45, label="Direct")
    plt.scatter(df["RUL_cycles"], df["RUL_pred_hybrid"], alpha=0.45, label="Hybrid")
    min_v = float(min(df["RUL_cycles"].min(), df["RUL_pred_direct"].min(), df["RUL_pred_hybrid"].min()))
    max_v = float(max(df["RUL_cycles"].max(), df["RUL_pred_direct"].max(), df["RUL_pred_hybrid"].max()))
    plt.plot([min_v, max_v], [min_v, max_v], "k--")
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("RUL Prediction vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_distribution(df: pd.DataFrame) -> None:
    err = df["RUL_pred_direct"] - df["RUL_cycles"]
    plt.figure(figsize=(8, 4))
    plt.hist(err, bins=35, alpha=0.8, color="#ff7f11")
    plt.xlabel("Prediction Error (cycles)")
    plt.ylabel("Count")
    plt.title("Error Distribution (Direct RUL Model)")
    plt.tight_layout()
    plt.show()


def sensitivity_dataframe(base_row: pd.Series, predict_fn) -> pd.DataFrame:
    """Generate sensitivity table over temperature, DoD and C-rate."""

    rows = []
    for t in [20, 25, 30, 35, 40, 45]:
        r = base_row.copy()
        r["avg_temp"] = t
        rows.append({"factor": "Temperature", "level": t, "rul": float(predict_fn(r))})

    for d in [0.30, 0.45, 0.60, 0.75, 0.90]:
        r = base_row.copy()
        r["dod"] = d
        rows.append({"factor": "DoD", "level": d, "rul": float(predict_fn(r))})

    for c in [0.5, 0.75, 1.0, 1.5, 2.0]:
        r = base_row.copy()
        r["c_rate"] = c
        rows.append({"factor": "C-rate", "level": c, "rul": float(predict_fn(r))})

    return pd.DataFrame(rows)


def plot_sensitivity(sens_df: pd.DataFrame) -> None:
    for factor in ["Temperature", "DoD", "C-rate"]:
        sub = sens_df[sens_df["factor"] == factor]
        plt.figure(figsize=(6.5, 3.8))
        plt.plot(sub["level"], sub["rul"], marker="o")
        plt.title(f"Sensitivity: {factor} vs Predicted RUL")
        plt.xlabel(factor)
        plt.ylabel("Predicted RUL (cycles)")
        plt.tight_layout()
        plt.show()


def explain_model_with_shap(model, X_sample: pd.DataFrame, max_display: int = 15) -> None:
    """Render SHAP summary plot; fallback to model feature importances if SHAP is unavailable."""

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, max_display=max_display)
    except Exception:
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=X_sample.columns).sort_values(ascending=False).head(max_display)
            plt.figure(figsize=(7, 4.5))
            imp.sort_values().plot(kind="barh")
            plt.title("Feature importance fallback (SHAP unavailable)")
            plt.tight_layout()
            plt.show()
