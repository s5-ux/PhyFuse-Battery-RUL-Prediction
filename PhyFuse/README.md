# EV Battery RUL Prediction for e-LCV Packs

Predicts battery failure before it happens using physics-informed AI.

## Overview
This project estimates:
- Remaining cycle life (`RUL_cycles`)
- Remaining calendar life (`RUL_months`)
- Remaining usable energy (`RUL_energy_kwh`)

The pipeline combines data-driven models with physics-inspired degradation features for EV battery packs under Indian operating conditions.

## Sensitivity Analysis
Sensitivity analysis is included to show how predicted battery life changes with key degradation drivers.

### i. Temperature Impact
- Variable: `avg_temp` (typically 20C to 45C)
- Method: Hold other features fixed and sweep temperature levels.
- Output: Predicted `RUL_mean` per temperature level.
- Expected trend: Higher temperature increases calendar stress and lowers RUL.

### ii. Depth of Discharge (DoD) Impact
- Variable: `dod` (for example 0.30 to 0.90)
- Method: Hold other features fixed and sweep DoD levels.
- Output: Predicted `RUL_mean` per DoD level.
- Expected trend: Higher DoD increases cycle stress and reduces battery life.

### iii. Charging Rate (C-rate) Impact
- Variable: `c_rate` (for example 0.5 to 2.0)
- Method: Hold other features fixed and sweep charging rate levels.
- Output: Predicted `RUL_mean` per C-rate level.
- Expected trend: Faster charging generally increases degradation and lowers RUL.

## Where It Is Implemented
- Dashboard sensitivity plots: `app.py`
- Feature physics terms (`cycle_stress`, `calendar_stress`, `total_degradation`): `feature_engineering.py`
- Hybrid prediction and uncertainty: `model_training.py`

## Run
1. Train artifacts:
   - `python model_training.py`
2. Launch dashboard:
   - `streamlit run app.py`

## Notes
- End-of-life (EOL) is defined as 80% capacity (SOH).
- Uncertainty is reported using lower and upper quantile bounds plus a confidence score.
