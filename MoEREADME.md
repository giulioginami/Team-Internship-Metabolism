# MoE-EDES: Mixture-of-Experts Personalised Diabetes Parameter Estimation

## Overview

This project implements a complete end-to-end pipeline for personalised diabetes parameter
estimation using the EDES (Eindhoven Diabetes Education Simulator) physiological model.
A Mixture-of-Experts (MoE) framework blends three population-specific PID controllers,
guided by a neural gating network trained on sparse OGTT observations.

---

## Pipeline — Run in This Order

| Step | Script | Language | Output |
|------|--------|----------|--------|
| 1 | `Generate_VirtualPopulation.m` | MATLAB | `virtual_population.mat` |
| 2 | `Label_VirtualPopulation.m` | MATLAB | `virtual_population_labelled.mat` |
| 3 | `Create_SparseDatasets.m` | MATLAB | `virtual_population_sparse.mat` |
| 4 | `PID_optimization_full.m` | MATLAB | Optimised k5/k6/k8 (hardcode into step 5) |
| 5 | `Save_PID_Predictions.m` | MATLAB | `pid_predictions.mat` |
| 6 | `python gating_network.py` | Python | `gating_network.pt`, `gating_weights.mat` |
| 7 | `final_architecture.m` | MATLAB | Personalised fit plot |

---

## Prerequisites

### MATLAB Toolboxes
- **Optimization Toolbox** — for `lsqnonlin`
- **Statistics and Machine Learning Toolbox** — for `lhsdesign`

### MATLAB Model Files (not included)
- `EDES_ODE.m` — the 5-state ODE right-hand side
- `integratorfunG.m` — ODE output function for the integral state

### Python Dependencies
```bash
pip install torch numpy h5py matplotlib scipy fpdf2
```

---

## File Descriptions

### Section 1 — Synthetic Data

#### `Generate_VirtualPopulation.m`
Generates a virtual population of N=5000 candidate patients.

- **Sampling**: Latin Hypercube Sampling (LHS) over 7 parameters — k1, k5, k6, k8, G_b, I_PL_b, BW
- **Simulation**: each candidate is run through the EDES ODE (`ode45`) for 0–480 min under a 75 g OGTT
- **Quality control**: rejects trajectories with negative values, out-of-range peaks, or oscillations
- **Noise**: multiplicative Gaussian noise — 5% on glucose, 10% on insulin

**Output struct fields** (`virtual_population.mat`):
| Field | Shape | Description |
|-------|-------|-------------|
| `time` | [1 x 481] | Full time vector (0 to 480 min) |
| `glucose_clean` | [n x 481] | Noise-free glucose trajectories |
| `insulin_clean` | [n x 481] | Noise-free insulin trajectories |
| `glucose_noisy` | [n x 481] | Glucose + 5% noise |
| `insulin_noisy` | [n x 481] | Insulin + 10% noise |
| `param_matrix` | [n x 7] | [k1, k5, k6, k8, G_b, I_PL_b, BW] per patient |

---

#### `Label_VirtualPopulation.m`
Classifies every virtual patient by ADA 2026 criteria applied to the 5-point OGTT
(t = 0, 30, 60, 90, 120 min). Splits the population into three dataset structs:
`dataset_NGT`, `dataset_IGT`, `dataset_T2DM`.

**Saves**: `virtual_population_labelled.mat`

---

#### `Classify_Diabetes_2H_OGTT.m`
Function implementing the ADA classification logic.

| Category | Fasting glucose G_0 | 2-h glucose G_2h | Extra filters |
|----------|---------------------|------------------|---------------|
| NGT | < 7.0 mmol/L | < 7.8 mmol/L | Peak G <= 10.0 |
| IGT | < 7.0 mmol/L | 7.8 – 11.1 mmol/L | Min G > 2.0 |
| T2DM | >= 7.0 mmol/L | >= 11.1 mmol/L | G_0 < 11.0, Peak G < 25.0 |

Patients satisfying none of the rules are placed in an out-of-distribution (excluded) group.

---

#### `Create_SparseDatasets.m`
Subsamples each labelled dataset to 7 clinically realistic OGTT time points:

```
t_sparse = [0, 30, 60, 90, 120]  minutes
```

**Saves**: `virtual_population_sparse.mat`  
**Dataset struct fields**: same as the labelled datasets but glucose/insulin are `[n x 5]`

---

### Section 2 — PID Optimization

#### `PID_optimization_full.m`
Optimises three sets of PID parameters (k5, k6, k8), one per population, by fitting the
full EDES ODE to the population median trajectory.

- **Representative patient**: uses median G_b, I_PL_b, BW, k1 across the population
- **Solver**: `lsqnonlin` with normalised residuals
  ```
  res = [(G_sim - G_med) / norm(G_med) ;
         (I_sim - I_med) / norm(I_med)]
  ```
- **Bounds**: k5 in [0, 0.17], k6 in [0, 0.34], k8 in [0, 10.0]

**Optimised expert parameters** (hardcoded in subsequent scripts):

| Expert | k5    | k6    | k8    |
|--------|-------|-------|-------|
| NGT    | 0.092 | 0.079 | 7.394 |
| IGT    | 0.006 | 0.089 | 4.724 |
| T2DM   | 0.014 | 0.000 | 5.755 |

---

### Section 3 — Gating Network

#### `Save_PID_Predictions.m`
For every patient in all three populations, runs the EDES ODE **three times** (once per
expert) and stores the predicted glucose and insulin at the sparse time points.

This precomputation lets the gating network train without any ODE calls in Python.

**Output** (`pid_predictions.mat`):
| Variable | Shape | Description |
|----------|-------|-------------|
| `G_pids` | [N x 3 x 5] | Expert glucose predictions at sparse times |
| `I_pids` | [N x 3 x 5] | Expert insulin predictions at sparse times |
| `labels` | [N x 1] | Ground-truth class: 1=NGT, 2=IGT, 3=T2DM |

Note: saved as `-v7.3` (HDF5). When loaded in Python via h5py, transpose arrays
because MATLAB stores in column-major order.

---

#### `gating_network.py`
Trains the MoE gating network and exports weights for MATLAB inference.

**Architecture:**
```
Input [10]  →  Linear(14→32)  →  ReLU
            →  Linear(32→32)  →  ReLU
            →  Linear(32→3)   →  Softmax
Output [3]   =  [w_NGT, w_IGT, w_T2DM]
```

**Input features**: `x = [G_sparse | I_sparse]` — 10 values standardised to zero mean
and unit variance using training-set statistics (X_mean, X_std).

**Loss**:
```
L = MSE(G_pred, G_obs) / Var(G) + MSE(I_pred, I_obs) / Var(I)
```
where `G_pred = w_NGT*G_NGT + w_IGT*G_IGT + w_T2DM*G_T2DM`.

**Training**: Adam, lr=1e-3, 300 epochs, batch 64, stratified 80/20 split.

**Outputs**:
- `gating_network.pt` — PyTorch model state dict
- `gating_weights.mat` — weights exported for MATLAB (W1, b1, W2, b2, W3, b3, X_mean, X_std)

---

### Section 4 — Mixture of Experts (full pipeline)

#### `final_architecture.m`
Applies the full MoE pipeline to a single patient.

**Steps:**

1. **Load patient data** — from `virtual_population_sparse.mat`

2. **Gating forward pass** (pure MATLAB, no Python needed):
   ```matlab
   x_norm = (x_in - X_mean) ./ X_std;
   h1 = max(0, W1 * x_norm' + b1);      % ReLU
   h2 = max(0, W2 * h1      + b2);      % ReLU
   z  = W3 * h2 + b3;
   w  = double(exp(z-max(z)) / sum(exp(z-max(z))));  % Softmax
   ```

3. **lsqnonlin optimisation** of `[k1, k5]`:
   - k6, k8 stay expert-specific (from `pids` matrix)
   - Initial k1: patient value or 0.028 for real data
   - Initial k5: `w' * pids(:,1)` — gating-weighted average
   - Bounds: k1 in [0.01, 0.05], k5 in [0, 0.17]

4. **Plot**: glucose and insulin panels showing
   - Dashed coloured curves: three expert simulations
   - Black solid curve: MoE weighted prediction
   - Black dots: observed sparse data

---

## Data Files Summary

| File | Format | Created by | Contents |
|------|--------|------------|----------|
| `virtual_population.mat` | v7.3 | Step 1 | Full virtual population |
| `virtual_population_labelled.mat` | v7.3 | Step 2 | NGT/IGT/T2DM split datasets |
| `virtual_population_sparse.mat` | v7.3 | Step 3 | 5-point sparse datasets |
| `pid_predictions.mat` | v7.3 | Step 5 | Expert predictions [N x 3 x 5] |
| `gating_network.pt` | PyTorch | Step 6 | Trained model weights |
| `gating_weights.mat` | v5 | Step 6 | Weights for MATLAB inference |

---

## Important Notes

- **Global variables**: `t_saved` and `G_PL_saved` are required by `EDES_ODE.m` via
  `integratorfunG.m`. They must be reset before every `ode45` call:
  ```matlab
  global t_saved G_PL_saved
  t_saved    = 0;
  G_PL_saved = G_b;
  ```
- **HDF5 transpose**: MATLAB saves v7.3 arrays in column-major order. When loading with
  h5py in Python, always call `.T` on the result to get row-major `[n x m]` arrays.
- **float32 vs double**: gating weights exported from PyTorch are float32. In MATLAB,
  always cast the gating output with `double(...)` before passing to `lsqnonlin`.
- **param_matrix columns**: `[k1, k5, k6, k8, G_b, I_PL_b, BW]` — column order is
  consistent across all scripts.
