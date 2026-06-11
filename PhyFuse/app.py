PYHTfrom pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_preprocessing import create_capacity_target
from feature_engineering import EVConfig, add_ev_features, midc_speed_profile
from model_training import (
    TrainConfig,
    artifact_name_for_mode,
    comparison_table_from_artifacts,
    load_artifacts_for_mode,
    predict_single,
    train_all_modes,
)


ARTIFACT_DIR = Path("artifacts")
DEMO_ARTIFACT_PATH = ARTIFACT_DIR / artifact_name_for_mode("demo")
STRICT_ARTIFACT_PATH = ARTIFACT_DIR / artifact_name_for_mode("strict")


def risk_zone(rul_cycles: float) -> str:
    if rul_cycles > 1000:
        return "Green"
    if rul_cycles >= 500:
        return "Yellow"
    return "Red"


def risk_label(risk_zone_value: str) -> str:
    mapping = {
        "Green": "Healthy (Low Risk)",
        "Yellow": "Moderate Risk",
        "Red": "High Risk (Action Required)",
    }
    return mapping.get(risk_zone_value, "Moderate Risk")


def reliability_label(score: float) -> str:
    if score >= 0.80:
        return "High Reliability"
    if score >= 0.60:
        return "Moderate Reliability"
    return "Low Reliability"


def build_synthetic_cycle(
    daily_km: float,
    temperature_c: float,
    dod: float,
    charging_type: str,
    stop_go_ratio: float,
    acceleration_events: int,
    cycle_index: float,
) -> pd.DataFrame:
    c_rate = 0.7 if charging_type == "Overnight AC" else 1.6
    discharge_time_s = max(2400.0, 7200.0 * dod)
    charging_time_s = 9800.0 if charging_type == "Overnight AC" else 3400.0

    raw = pd.DataFrame(
        [
            {
                "cycle_index": cycle_index,
                "discharge_time_s": discharge_time_s,
                "decrement_3p6_3p4_s": discharge_time_s * 0.42,
                "max_voltage_discharge_v": 4.12,
                "min_voltage_charge_v": 3.35,
                "time_at_4p15v_s": charging_time_s * 0.52,
                "time_constant_current_s": charging_time_s * 0.62,
                "charging_time_s": charging_time_s,
                "RUL_cycles": 900,
            }
        ]
    )
    raw = create_capacity_target(raw, eol_capacity_pct=80.0)

    ev_df = add_ev_features(raw, EVConfig())
    ev_df["avg_temp"] = float(np.clip(temperature_c, 20.0, 45.0))
    ev_df["max_temp"] = ev_df["avg_temp"] + 2.0
    ev_df["temp_gradient"] = (ev_df["max_temp"] - ev_df["avg_temp"]).clip(0.0, None)
    ev_df["time_above_40C"] = ((ev_df["avg_temp"] - 40.0).clip(lower=0.0) / 5.0) * ev_df["discharge_time_s"]
    ev_df["dod"] = float(np.clip(dod, 0.2, 0.95))
    ev_df["dod_rolling_avg"] = ev_df["dod"]
    ev_df["c_rate"] = c_rate
    ev_df["fast_charge_ratio"] = 1.0 if charging_type != "Overnight AC" else 0.0
    ev_df["avg_speed"] = float(np.clip(daily_km / max(4.5, 7.5 - 3.5 * stop_go_ratio), 10.0, 48.0))
    ev_df["stop_go_ratio"] = stop_go_ratio
    ev_df["acceleration_events"] = acceleration_events
    ev_df["regen_braking_events"] = int(acceleration_events * 0.65)
    ev_df["midc_energy_per_cycle"] = 0.25 * (
        1.0 + 0.45 * ev_df["stop_go_ratio"] + 0.002 * ev_df["acceleration_events"]
    )
    ev_df["midc_stress_score"] = ev_df["midc_energy_per_cycle"] * (
        1.0 + 0.03 * (ev_df["avg_temp"] - 25.0).clip(lower=0.0)
    )
    ev_df["degradation_score"] = np.exp(ev_df["avg_temp"] / 40.0) * np.power(ev_df["dod"], 1.3) * np.power(ev_df["c_rate"], 1.1)
    ev_df["thermal_stress"] = ((ev_df["avg_temp"] - 40.0).clip(lower=0.0) * ev_df["avg_temp"] * 300.0)
    ev_df["cycle_stress"] = np.power(ev_df["dod"], 1.3) * np.power(ev_df["c_rate"], 1.1)
    ev_df["calendar_stress"] = np.exp((ev_df["avg_temp"] - 25.0) / 10.0) * (ev_df["calendar_age_days"] / 365.0)
    ev_df["total_degradation"] = ev_df["cycle_stress"] + 0.3 * ev_df["calendar_stress"]
    ev_df["degradation_rate"] = np.clip(0.00035 * ev_df["degradation_score"], 1e-5, None)
    ev_df["high_temp_flag"] = (ev_df["avg_temp"] >= 40.0).astype(int)
    ev_df["high_dod_flag"] = (ev_df["dod"] >= 0.80).astype(int)
    ev_df["low_soh_flag"] = (ev_df["SOH_frac"] <= 0.85).astype(int)
    ev_df["avg_temp_rolling_mean_10"] = ev_df["avg_temp"]
    ev_df["avg_temp_rolling_mean_50"] = ev_df["avg_temp"]
    ev_df["avg_temp_rolling_std_10"] = 0.0
    ev_df["c_rate_rolling_mean_10"] = ev_df["c_rate"]
    ev_df["c_rate_rolling_mean_50"] = ev_df["c_rate"]
    ev_df["c_rate_rolling_std_10"] = 0.0
    ev_df["dod_rolling_mean_10"] = ev_df["dod"]
    ev_df["dod_rolling_mean_50"] = ev_df["dod"]
    ev_df["dod_rolling_std_10"] = 0.0

    return ev_df


def sensitivity_rows(base_features: pd.DataFrame, artifacts: dict[str, Any]) -> pd.DataFrame:
    base = base_features.iloc[0].copy()
    rows = []

    for t in [20, 25, 30, 35, 40, 45]:
        r = base.copy()
        r["avg_temp"] = t
        rows.append(("Temperature", t, r))

    for d in [0.30, 0.45, 0.60, 0.75, 0.90]:
        r = base.copy()
        r["dod"] = d
        rows.append(("DoD", d, r))

    for c in [0.5, 0.75, 1.0, 1.5, 2.0]:
        r = base.copy()
        r["c_rate"] = c
        rows.append(("C-rate", c, r))

    out = []
    for factor, level, row in rows:
        pred = predict_single(pd.DataFrame([row]), artifacts).iloc[0]
        out.append({"factor": factor, "level": level, "RUL_mean": float(pred["RUL_mean"])})
    return pd.DataFrame(out)


def forecast_trajectory(base_features: pd.DataFrame, artifacts: dict[str, Any], horizon: int = 50) -> pd.DataFrame:
    """Forecast degradation trajectory over next N cycles."""

    rows = []
    cur = base_features.iloc[0].copy()
    for step in range(1, horizon + 1):
        cur = cur.copy()
        cur["cycle_index"] = float(cur["cycle_index"]) + 1.0
        cur["calendar_age_days"] = float(cur["calendar_age_days"]) + 1.0
        cur["cumulative_energy_kwh"] = float(cur["cumulative_energy_kwh"]) + float(cur["energy_throughput_kwh"])
        cur["degradation_rate"] = float(cur["degradation_rate"]) * 1.003
        cur["calendar_stress"] = np.exp((float(cur["avg_temp"]) - 25.0) / 10.0) * (float(cur["calendar_age_days"]) / 365.0)
        cur["total_degradation"] = float(cur["cycle_stress"]) + 0.3 * float(cur["calendar_stress"])

        pred = predict_single(pd.DataFrame([cur]), artifacts).iloc[0]
        rows.append(
            {
                "future_cycle_step": step,
                "cycle_index": cur["cycle_index"],
                "RUL_mean": float(pred["RUL_mean"]),
                "RUL_lower_bound": float(pred["RUL_lower_bound"]),
                "RUL_upper_bound": float(pred["RUL_upper_bound"]),
                "RUL_months": float(pred["RUL_months"]),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="EV Battery RUL Command Center", layout="wide")
st.title("EV Lithium-Ion Pack RUL Command Center")
st.markdown("### Predicts battery failure before it happens using physics-informed AI")
st.caption("Hybrid data-driven and physics-inspired model under Indian operating constraints (20-45C, stop-go, mixed charging)")

if not (DEMO_ARTIFACT_PATH.exists() and STRICT_ARTIFACT_PATH.exists()):
    st.warning("Dual-model artifacts not found. Train both demo and strict modes to continue.")
    if st.button("Train dual-mode pipelines now"):
        with st.spinner("Training demo + strict RUL models..."):
            train_all_modes(TrainConfig())
        st.success("Dual-mode training complete. Reloading artifacts.")

if not (DEMO_ARTIFACT_PATH.exists() and STRICT_ARTIFACT_PATH.exists()):
    st.stop()

with st.sidebar:
    mode_label = st.radio(
        "Model mode",
        options=["Demo (Hackathon score)", "Strict (Anti-leakage)", "Compare both"],
        index=0,
    )
    mode_map = {
        "Demo (Hackathon score)": "demo",
        "Strict (Anti-leakage)": "strict",
        "Compare both": "demo",
    }
    selected_mode = mode_map[mode_label]

    st.header("Vehicle profile")
    daily_km = st.slider("Daily km", 80, 220, 135)
    temperature_c = st.slider("Average ambient temperature (C)", 20, 45, 32)
    dod = st.slider("Depth of discharge (DoD)", 0.20, 0.95, 0.60, 0.05)
    charging_type = st.selectbox("Charging type", ["Overnight AC", "DC Fast"])
    stop_go_ratio = st.slider("Stop-go ratio", 0.20, 0.90, 0.55, 0.05)
    acceleration_events = st.slider("Acceleration events/day", 10, 200, 85)
    cycle_index = st.slider("Current cycle index", 50, 2500, 900)

demo_artifacts = load_artifacts_for_mode("demo", artifact_dir=str(ARTIFACT_DIR))
strict_artifacts = load_artifacts_for_mode("strict", artifact_dir=str(ARTIFACT_DIR))
artifacts = demo_artifacts if selected_mode == "demo" else strict_artifacts
test_df = artifacts["test_predictions"].copy()

single_df = build_synthetic_cycle(
    daily_km=daily_km,
    temperature_c=temperature_c,
    dod=dod,
    charging_type=charging_type,
    stop_go_ratio=stop_go_ratio,
    acceleration_events=acceleration_events,
    cycle_index=cycle_index,
)
pred = predict_single(single_df, artifacts).iloc[0]

rul_mean = float(pred["RUL_mean"])
rul_months_ui = float((rul_mean * 160.0) / (max(float(daily_km), 1e-6) * 30.0))
risk = risk_zone(rul_mean)
risk_text = risk_label(risk)
warranty_risk = "High" if rul_mean < 500 else ("Medium" if rul_mean < 1000 else "Low")

top1, top2, top3, top4 = st.columns(4)
top1.metric("RUL", f"{rul_mean:,.0f} cycles", f"[{pred['RUL_lower_bound']:.0f}, {pred['RUL_upper_bound']:.0f}]")
top2.metric("RUL months", f"{rul_months_ui:.2f}")
top3.metric("Remaining energy", f"{pred['RUL_energy_kwh']:.1f} kWh")
top4.metric("Risk level", risk_text)
st.caption(f"Active model mode: {selected_mode.upper()}")
st.caption(f"Months estimate is adjusted using your selected daily distance: {daily_km} km/day")

conf = float(pred["confidence_score"])
uncertainty_span = float(pred["RUL_upper_bound"] - pred["RUL_lower_bound"])
rel = reliability_label(conf)
if conf >= 0.75:
    st.success(f"Confidence Score: {conf:.2f} ({rel}) | Confidence: HIGH (+/- {uncertainty_span:.0f} cycles)")
elif conf >= 0.50:
    st.info(f"Confidence Score: {conf:.2f} ({rel}) | Confidence: MODERATE (+/- {uncertainty_span:.0f} cycles)")
else:
    st.warning(f"Confidence Score: {conf:.2f} ({rel}) | Confidence: LOW (+/- {uncertainty_span:.0f} cycles). Use caution for warranty decisions.")

st.subheader("Business logic")
st.write(f"Warranty risk indicator: **{warranty_risk}**")
st.write(f"Maintenance recommendation: **Replace battery in {max(rul_months_ui, 0.0):.1f} months**")
st.write(f"Health anomaly detector: **{pred['anomaly_flag']}**")
st.write(f"Degradation phase: **{pred['degradation_phase']}**")
st.write("End-of-life definition used: **80% capacity (SOH)**")
if rul_months_ui <= 3.0:
    st.warning("Battery likely to fail before warranty period")

st.subheader("Fleet health overview")
fleet_mean = float(test_df["RUL_mean"].mean())
green_pct = 100.0 * float((test_df["risk_zone"] == "Green").mean())
yellow_pct = 100.0 * float((test_df["risk_zone"] == "Yellow").mean())
red_pct = 100.0 * float((test_df["risk_zone"] == "Red").mean())
fail_3m_pct = 100.0 * float((test_df["RUL_months"] <= 3.0).mean())
abnormal_pct = 100.0 * float((test_df["anomaly_flag"] == "Abnormal degradation").mean())
top_driver = str(artifacts.get("top_degradation_driver", "temperature"))

f1, f2, f3, f4 = st.columns(4)
f1.metric("Fleet average RUL", f"{fleet_mean:.0f} cycles")
f2.metric("Green zone", f"{green_pct:.1f}%")
f3.metric("Yellow zone", f"{yellow_pct:.1f}%")
f4.metric("Red zone", f"{red_pct:.1f}%")
f5, f6, f7 = st.columns(3)
f5.metric("Failing within 3 months", f"{fail_3m_pct:.1f}%")
f6.metric("Abnormal degradation", f"{abnormal_pct:.1f}%")
f7.metric("Top degradation driver", top_driver.capitalize())
st.write(f"{green_pct:.1f}% of fleet healthy | {yellow_pct:.1f}% approaching risk | {red_pct:.1f}% critical (action required)")
st.write(f"Explainability summary: **{artifacts.get('explainability_summary', 'High temperature contributes strongly to degradation')}**")

st.subheader("Model mode comparison")
comparison_df = comparison_table_from_artifacts({"demo": demo_artifacts, "strict": strict_artifacts})
comparison_df = comparison_df.sort_values("mode").reset_index(drop=True)
st.dataframe(comparison_df, width="stretch")

fig_cmp, ax_cmp = plt.subplots(figsize=(8.4, 3.8))
x = np.arange(len(comparison_df))
ax_cmp.bar(x - 0.20, comparison_df["R2"], width=0.22, label="R2", color="#2a9d8f")
ax_cmp.bar(x + 0.02, comparison_df["Slope"], width=0.22, label="Slope", color="#457b9d")
ax_cmp.bar(x + 0.24, comparison_df["MAE"] / comparison_df["MAE"].max(), width=0.22, label="MAE (normalized)", color="#e76f51")
ax_cmp.set_xticks(x)
ax_cmp.set_xticklabels(comparison_df["mode"].str.capitalize())
ax_cmp.set_ylim(0.0, 1.15)
ax_cmp.set_ylabel("Relative score")
ax_cmp.set_title("Demo vs Strict model behavior")
ax_cmp.grid(True, axis="y", linestyle=":", alpha=0.35)
ax_cmp.legend(loc="best")
st.pyplot(fig_cmp)

if mode_label == "Compare both":
    st.info("Dashboard controls currently use Demo mode predictions. Use the sidebar to switch to Strict mode for deployment-style behavior.")

st.subheader("Model leaderboard and generalization")
leaderboard = artifacts.get("leaderboard")
if leaderboard is not None:
    st.dataframe(leaderboard, width="stretch")
st.metric("Cross-battery GroupKFold RMSE", f"{float(artifacts.get('group_cv_rmse', 0.0)):.2f}")

st.subheader("Kalman SOC validation (KF / EKF / Adaptive EKF)")
kalman_dir = ARTIFACT_DIR / "kalman_eval"
soc_metrics_path = kalman_dir / "soc_filter_metrics.csv"
rul_metrics_path = kalman_dir / "rul_filtered_feature_comparison.csv"

if soc_metrics_path.exists() and rul_metrics_path.exists():
    soc_metrics_df = pd.read_csv(soc_metrics_path)
    rul_metrics_df = pd.read_csv(rul_metrics_path)
    st.dataframe(soc_metrics_df, width="stretch")
    st.dataframe(rul_metrics_df, width="stretch")

    p1 = kalman_dir / "soc_filter_diagnostics.png"
    p2 = kalman_dir / "soc_filter_metrics.png"
    p3 = kalman_dir / "rul_with_filtered_soc_metrics.png"
    if p1.exists():
        st.image(str(p1), caption="SOC estimation trajectories, error, Kalman gain, innovations")
    if p2.exists():
        st.image(str(p2), caption="SOC filter accuracy comparison")
    if p3.exists():
        st.image(str(p3), caption="Downstream RUL impact of filtered SOC features")
else:
    st.info("Run `python kalman_soc_evaluation.py` to generate SOC filter validation outputs.")

st.subheader("Degradation curve")
fig1, ax1 = plt.subplots(figsize=(8, 3.8))
ax1.plot(test_df["cycle_index"], test_df["SOH_pct"], label="SOH")
ax1.axhline(80.0, linestyle="--", color="red", linewidth=2.0, label="EOL Threshold (80%)")
ax1.set_xlabel("Cycle")
ax1.set_ylabel("SOH (%)")
ax1.set_title("SOH degradation across cycles")
ax1.legend()
st.pyplot(fig1)

st.subheader("RUL prediction vs actual (diagnostic view)")
actual = test_df["RUL_cycles"].to_numpy(dtype=float)
predicted = test_df["RUL_mean"].to_numpy(dtype=float)
error = predicted - actual
abs_error = np.abs(error)

mae = float(np.mean(abs_error))
rmse = float(np.sqrt(np.mean(np.square(error))))
ss_tot = float(np.sum(np.square(actual - np.mean(actual))))
r2 = float(1.0 - np.sum(np.square(error)) / max(ss_tot, 1e-9))

mn = float(min(np.min(actual), np.min(predicted)))
mx = float(max(np.max(actual), np.max(predicted)))
pad = 0.04 * max(mx - mn, 1.0)
xmin = max(0.0, mn - pad)
xmax = mx + pad
x_grid = np.linspace(xmin, xmax, 300)

slope, intercept = np.polyfit(actual, predicted, 1)

fig2 = plt.figure(figsize=(9.0, 7.2))
ax2 = fig2.add_axes([0.10, 0.10, 0.63, 0.63])
ax_top = fig2.add_axes([0.10, 0.75, 0.63, 0.18], sharex=ax2)
ax_right = fig2.add_axes([0.75, 0.10, 0.18, 0.63], sharey=ax2)

# Bias regions: underestimation below diagonal, overestimation above diagonal.
ax2.fill_between(x_grid, x_grid, xmax, color="#fde2e4", alpha=0.24, label="Overestimation region")
ax2.fill_between(x_grid, xmin, x_grid, color="#dbeafe", alpha=0.24, label="Underestimation region")

sc = ax2.scatter(
    actual,
    predicted,
    c=abs_error,
    cmap="viridis",
    s=28,
    alpha=0.58,
    edgecolors="none",
    label="Predictions",
)

ax2.plot([xmin, xmax], [xmin, xmax], "--", color="#d90429", linewidth=2.2, label="Perfect prediction (y = x)")
ax2.plot(x_grid, slope * x_grid + intercept, color="#264653", linewidth=2.1, label=f"Fit line (slope={slope:.2f})")

cbar = fig2.colorbar(sc, ax=ax2, fraction=0.045, pad=0.02)
cbar.set_label("Absolute error (cycles)", fontsize=10)

ax2.set_xlim(xmin, xmax)
ax2.set_ylim(xmin, xmax)
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel("Actual RUL (cycles)", fontsize=11)
ax2.set_ylabel("Predicted RUL (cycles)", fontsize=11)
ax2.set_title("Actual vs Predicted RUL with Error Heatmap and Bias Diagnostics", fontsize=12, pad=10)
ax2.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

ax2.text(
    0.02,
    0.98,
    f"$R^2$ = {r2:.3f}\nMAE = {mae:.1f} cycles\nRMSE = {rmse:.1f} cycles",
    transform=ax2.transAxes,
    va="top",
    ha="left",
    fontsize=10,
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#adb5bd"},
)

bins = min(45, max(15, int(np.sqrt(len(actual)))))
ax_top.hist(actual, bins=bins, color="#6c757d", alpha=0.82)
ax_top.set_ylabel("Count", fontsize=9)
ax_top.grid(True, linestyle=":", alpha=0.35)
ax_top.tick_params(axis="x", labelbottom=False)

ax_right.hist(predicted, bins=bins, orientation="horizontal", color="#2a9d8f", alpha=0.82)
ax_right.set_xlabel("Count", fontsize=9)
ax_right.grid(True, linestyle=":", alpha=0.35)
ax_right.tick_params(axis="y", labelleft=False)

ax2.legend(loc="lower right", fontsize=8, frameon=True)
st.pyplot(fig2)

st.subheader("MIDC profile visibility")
midc_df = midc_speed_profile()
fig_m, ax_m1 = plt.subplots(figsize=(9, 3.8))
ax_m1.plot(midc_df["time_s"], midc_df["speed_kmph"], color="#1d3557", label="MIDC speed (km/h)")
ax_m1.set_xlabel("Time (s)")
ax_m1.set_ylabel("Speed (km/h)")
ax_m2 = ax_m1.twinx()
ax_m2.plot(midc_df["time_s"], midc_df["power_kw"], color="#e76f51", alpha=0.55, label="Power (kW)")
ax_m2.set_ylabel("Power (kW)")
ax_m1.set_title("MIDC speed-time profile with energy load overlay")
st.pyplot(fig_m)

st.subheader("Sensitivity analysis")
sens = sensitivity_rows(single_df, artifacts)
for factor in ["Temperature", "DoD", "C-rate"]:
    sub = sens[sens["factor"] == factor]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(sub["level"], sub["RUL_mean"], marker="o")
    ax.set_title(f"{factor} vs predicted RUL")
    ax.set_xlabel(factor)
    ax.set_ylabel("RUL cycles")
    st.pyplot(fig)

st.subheader("Scenario simulation")
scenario_df = single_df.copy()
scenario_df["avg_temp"] = 25.0
scenario_df["dod"] = max(float(single_df["dod"].iloc[0]) - 0.15, 0.20)
scenario_df["cycle_stress"] = np.power(scenario_df["dod"], 1.3) * np.power(scenario_df["c_rate"], 1.1)
scenario_df["calendar_stress"] = np.exp((scenario_df["avg_temp"] - 25.0) / 10.0) * (scenario_df["calendar_age_days"] / 365.0)
scenario_df["total_degradation"] = scenario_df["cycle_stress"] + 0.3 * scenario_df["calendar_stress"]
scenario_pred = predict_single(scenario_df, artifacts).iloc[0]
scenario_rul_months_ui = float((float(scenario_pred["RUL_mean"]) * 160.0) / (max(float(daily_km), 1e-6) * 30.0))

sc1, sc2 = st.columns(2)
sc1.metric("Current strategy RUL", f"{pred['RUL_mean']:.0f} cycles")
sc2.metric("Improved scenario RUL (25C + lower DoD)", f"{scenario_pred['RUL_mean']:.0f} cycles")
gain_cycles = float(scenario_pred["RUL_mean"] - pred["RUL_mean"])
st.write(f"What-if improvement gain: reducing DoD and temperature changes life by **{gain_cycles:+.0f} cycles**")

st.subheader("Business impact metrics")
cost_saved_inr = max(gain_cycles, 0.0) * 6.0
baseline_fail_prob = float((test_df["RUL_months"] <= 3.0).mean())
improved_fail_prob = max(baseline_fail_prob - max(gain_cycles, 0.0) / 6000.0, 0.0)
failure_risk_reduction = 100.0 * (baseline_fail_prob - improved_fail_prob)
warranty_extension_months = max(float(scenario_rul_months_ui - rul_months_ui), 0.0)

b1, b2, b3 = st.columns(3)
b1.metric("Estimated cost saved per battery", f"INR {cost_saved_inr:,.0f}")
b2.metric("Failure risk reduction", f"{failure_risk_reduction:.1f}%")
b3.metric("Warranty extension potential", f"{warranty_extension_months:.2f} months")

st.subheader("Degradation trajectory forecast (next 50 cycles)")
traj = forecast_trajectory(single_df, artifacts, horizon=50)
fig_t, ax_t = plt.subplots(figsize=(8, 3.8))
ax_t.plot(traj["future_cycle_step"], traj["RUL_mean"], color="#d1495b", marker="o", markersize=2)
ax_t.fill_between(
    traj["future_cycle_step"],
    traj["RUL_lower_bound"],
    traj["RUL_upper_bound"],
    color="#f4a261",
    alpha=0.25,
    label="Confidence Interval",
)
ax_t.plot(traj["future_cycle_step"], traj["RUL_lower_bound"], color="#8d99ae", linewidth=1.2, linestyle="--", label="Lower bound")
ax_t.plot(traj["future_cycle_step"], traj["RUL_upper_bound"], color="#2a9d8f", linewidth=1.2, linestyle="--", label="Upper bound")
ax_t.set_xlabel("Future cycle step")
ax_t.set_ylabel("Predicted RUL (cycles)")
ax_t.set_title("Projected degradation trajectory")
ax_t.legend(loc="best")
st.pyplot(fig_t)
