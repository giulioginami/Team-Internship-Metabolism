# Frontend — Interactive MoE-EDES Explorer

`edes_moe_frontend.py` is an interactive [Dash](https://dash.plotly.com/) app that
puts a live, browser-based layer on top of this project's Mixture-of-Experts EDES
pipeline. It is a pure-Python companion to the MATLAB scripts — no MATLAB needed to run it.

## What it shows

- **Patient selector** — any of the 118 real Japanese OGTT patients from
  `EDES_MoE/Datasets/Real Dataset/japan_population_labelled.mat` (or a manual baseline).
- **Gating network** — runs the trained network
  (`EDES_MoE/Gating Network/gating_weights.mat`) as a pure-Python forward pass on the
  patient's sparse OGTT `[G(0,30,60,90,120) | I(...)]` and shows the `[w_NGT, w_IGT, w_T2DM]`
  expert weights, exactly as `Fit_MoE.m` §2.
- **k1 / k5 sliders** — the two MoE-personalised parameters. `k6` and `k8` stay
  expert-specific (NGT/IGT/T2DM values from `PID_optimization.m`) and are blended by the
  gating weights. All other EDES parameters are fixed to the Rozendaal 2018 baseline.
- **Plots** — the three expert EDES simulations (dotted), the MoE weighted prediction
  (black), the observed sparse points, the population reference shadow (IQR + median) for
  the patient's ADA category, and the P/I/D contributions to pancreatic insulin.

The EDES model is a faithful explicit-Euler port of `EDES_PID/EDES_ODE.m`
(`EDES_Parameters.m` / `EDES_Initial.m` constants).

## Run

```bash
pip install dash dash-bootstrap-components plotly numpy scipy
python Frontend/edes_moe_frontend.py
# open http://127.0.0.1:8050
```

If the dataset / gating files are missing, the app still launches in **manual mode**
(custom fasting values, equal expert weights).
