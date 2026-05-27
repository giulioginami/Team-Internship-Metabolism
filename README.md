# MoE-EDES: Mixture-of-Experts Personalised Diabetes Parameter Estimation

## Overview

This project implements a complete end-to-end pipeline for personalised diabetes parameter
estimation using the EDES (Eindhoven Diabetes Education Simulator) physiological model.
A Mixture-of-Experts (MoE) framework blends three population-specific EDES experts,
guided by a neural gating network trained on sparse OGTT observations.

The fitted parameters — primarily **k1** (gastric emptying rate) and **k5** (insulin-dependent
glucose uptake rate) — allow the EDES model to replicate an individual patient's
postprandial glucose and insulin dynamics from a standard 75 g 5-point OGTT.

The MoE approach is compared against a **single-expert baseline**, and both are evaluated on a real-world dataset of 118 Japanese patients with OGTT measurements and a gold-standard GIR from euglycemic hyperinsulinaemic clamp.

---

## Repository Structure

The project is organised into two top-level folders (`EDES_MoE/` and `EDES_PID/`) plus
a small number of root-level files.

```
(root)
  main.m                              Run both MoE and single-expert on one patient and compare
  startup.m                           MATLAB startup script (adds subfolders to path)
  README.md

EDES_MoE/
  Datasets/
    Real Dataset/
      japan_population_labelled.mat   118 real patients: glucose, insulin, BW, GIR (external)

    Virtual Dataset/
      Generate_VirtualPopulation.m    Generate N=5000 virtual patients via LHS + EDES ODE
      Label_VirtualPopulation.m       Classify patients by ADA 2026 criteria (NGT/IGT/T2DM)
      Classify_Diabetes_2H_OGTT.m     ADA classification logic (helper function)
      Create_SparseDatasets.m         Subsample to 5-point OGTT (t = 0,30,60,90,120 min)
      Calculate_Matsuda_5_OGTT.m      Compute Matsuda insulin-sensitivity index from 5-pt OGTT
      Calculate_QUICKI.m              Compute QUICKI insulin-sensitivity index
      virtual_population.mat          Full virtual population (n × 481 trajectories)
      virtual_population_labelled.mat NGT/IGT/T2DM split with ADA labels
      virtual_population_sparse.mat   5-point sparse OGTT datasets

  Gating Network/
    gating_network.py                 Train neural gating network; export weights to MATLAB
    gating_network.pt                 Trained PyTorch model state dict
    gating_weights.mat                Weights exported for MATLAB inference

  Mixture of Experts/
    Fit_MoE.m                         MoE fit on one real patient from the Japan dataset
    Dataset_Fit_MoE.m                 MoE fit on all 118 Japan dataset patients (population evaluation)
    Fit_PID.m                         Single-expert fit on one real patient
    Dataset_Fit_PID.m                 Single-expert fit on all 118 patients (population evaluation)

  PID Optimization/
    PID_optimization.m                Optimise expert PID parameters (k5, k6, k8) per category
    Save_PID_Predictions.m            Pre-compute expert EDES predictions for gating network training

EDES_PID/                             Shared EDES core functions and utilities (not modified)
  EDES_ODE.m                          5-state ODE right-hand side
  EDES_Parameters.m                   Map optimised parameters to full 15-parameter vector
  EDES_Initial.m                      Compute initial conditions and model constants
  EDES_ErrorFunc.m                    Error function for single-expert lsqnonlin
  Fit_EDES.m                          Single lsqnonlin call (Shauna)
  Fit_EDES_LatinHyperCube.m           LHS multi-start wrapper around Fit_EDES (Shauna)
  integratorfunG.m                    ODE output function for integral state (global variables)
  Simulate_EDES.m                     Simulate EDES given a parameter set and return trajectories
  Plot_EDES.m                         Plot a single EDES simulation (glucose + insulin)
  Plot_MultiFit_EDES.m                Plot multiple EDES fits overlaid
  Script_fit_and_simulated_EDES.m     Example script: fit EDES to sample data and simulate
  sample_data.mat                     Example OGTT data for standalone EDES fitting/testing
```

---

## Full Pipeline — Run in This Order

| Step | Script | Language | Output |
|------|--------|----------|--------|
| 1 | `startup.m` | MATLAB | none |
| 2 | `Generate_VirtualPopulation.m` | MATLAB | `virtual_population.mat` |
| 3 | `Label_VirtualPopulation.m` | MATLAB | `virtual_population_labelled.mat` |
| 4 | `Create_SparseDatasets.m` | MATLAB | `virtual_population_sparse.mat` |
| 5 | `PID_optimization.m` | MATLAB | Optimised k5/k6/k8 (hardcode into step 5) |
| 6 | `Save_PID_Predictions.m` | MATLAB | `pid_predictions.mat` |
| 7 | `python gating_network.py` | Python | `gating_network.pt`, `gating_weights.mat` |
| 8 | `Fit_MoE.m` | MATLAB | MoE personalised fit (one patient) |
| 9 | `Dataset_Fit_MoE.m` | MATLAB | MoE population evaluation (118 patients) |

---

## Prerequisites

### MATLAB Toolboxes
- **Optimization Toolbox** — for `lsqnonlin`
- **Statistics and Machine Learning Toolbox** — for `lhsdesign`

### Python Dependencies
```bash
pip install torch numpy h5py matplotlib scipy fpdf2
```

---

## Section 1 — Synthetic Data Generation

### `Generate_VirtualPopulation.m`

Generates a virtual population of N=5000 candidate patients.

- **Sampling**: Latin Hypercube Sampling (LHS) over 7 parameters — k1, k5, k6, k8, G_b, I_PL_b, BW
- **Simulation**: each candidate is run through the EDES ODE (`ode45`) for 0–480 min under a 75 g OGTT
- **Quality control**: rejects trajectories with negative values, out-of-range peaks, or oscillations
- **Noise**: noise to replicate realistic measurements, 5% on glucose, 10% on insulin

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

### `Label_VirtualPopulation.m`

Classifies every virtual patient by ADA 2026 criteria applied to the 5-point OGTT
(t = 0, 30, 60, 90, 120 min). Splits the population into three dataset structs:
`dataset_NGT`, `dataset_IGT`, `dataset_T2DM`.

**Saves**: `virtual_population_labelled.mat`

---

### `Classify_Diabetes_2H_OGTT.m`

Helper function implementing the ADA classification logic.

| Category | Fasting glucose G_0 | 2-h glucose G_2h | Extra filters |
|----------|---------------------|------------------|---------------|
| NGT | < 7.0 mmol/L | < 7.8 mmol/L | Peak G ≤ 10.0 |
| IGT | < 7.0 mmol/L | 7.8 – 11.1 mmol/L | Min G > 2.0 |
| T2DM | ≥ 7.0 mmol/L | ≥ 11.1 mmol/L | G_0 < 11.0, Peak G < 25.0 |

Patients satisfying none of the rules are placed in an out-of-distribution (excluded) group.

---

### `Create_SparseDatasets.m`

Subsamples each labelled dataset to 5 clinically realistic OGTT time points:

```
t_sparse = [0, 30, 60, 90, 120]  minutes
```

**Saves**: `virtual_population_sparse.mat`  
**Dataset struct fields**: same as the labelled datasets but glucose/insulin are `[n x 5]`.

---

### `Calculate_Matsuda_5_OGTT.m`

Computes the Matsuda insulin-sensitivity index from 5-point OGTT glucose and insulin data.
Used as an auxiliary metabolic biomarker alongside the main pipeline.

---

### `Calculate_QUICKI.m`

Computes the QUICKI (Quantitative Insulin Sensitivity Check Index) from fasting glucose
and insulin values. Used as an auxiliary insulin-sensitivity biomarker.

---

## Section 2 — PID Optimization

### `PID_optimization.m`

Optimises three sets of PID parameters (k5, k6, k8), one per population, by fitting the
EDES ODE to the population median trajectory.

- **Representative patient**: uses median G_b, I_PL_b, BW, k1 across the population
- **Solver**: `lsqnonlin` with normalised residuals:
  ```
  res = [(G_sim - G_med) / norm(G_med);
         (I_sim - I_med) / norm(I_med)]
  ```
- **Bounds**: k5 ∈ [0, 0.17], k6 ∈ [0, 0.34], k8 ∈ [0, 10.0]

**Optimised expert parameters** (hardcoded in all subsequent scripts):

| Expert | k5    | k6    | k8    |
|--------|-------|-------|-------|
| NGT    | 0.092 | 0.079 | 7.394 |
| IGT    | 0.006 | 0.089 | 4.724 |
| T2DM   | 0.014 | 0.000 | 5.755 |

---

### `Save_PID_Predictions.m`

For every patient in all three populations, runs the EDES ODE **three times** (once per
expert) and stores the predicted glucose and insulin at the sparse time points.
This precomputation lets the gating network train without ODE calls in Python.

**Output** (`pid_predictions.mat`):

| Variable | Shape | Description |
|----------|-------|-------------|
| `G_pids` | [N x 3 x 5] | Expert glucose predictions at sparse times |
| `I_pids` | [N x 3 x 5] | Expert insulin predictions at sparse times |
| `labels` | [N x 1] | Ground-truth class: 1=NGT, 2=IGT, 3=T2DM |

> Saved as `-v7.3` (HDF5). When loading with h5py in Python, call `.T` on the result
> because MATLAB stores arrays in column-major order.

---

## Section 3 — Gating Network

### `gating_network.py`

Trains the MoE gating network and exports weights for MATLAB inference.

**Architecture:**
```
Input [10]  →  Linear(10→32)  →  ReLU
            →  Linear(32→32)  →  ReLU
            →  Linear(32→3)   →  Softmax
Output [3]  =  [w_NGT, w_IGT, w_T2DM]
```

**Input features**: `x = [G_sparse | I_sparse]` — 10 values standardised to zero mean
and unit variance using training-set statistics (X_mean, X_std).

**Loss**:
```
L = MSE(G_pred, G_obs) / Var(G) + MSE(I_pred, I_obs) / Var(I)
```
where `G_pred = w_NGT * G_NGT + w_IGT * G_IGT + w_T2DM * G_T2DM`.

**Training**: Adam, lr=1e-3, 300 epochs, batch 64, stratified 80/20 split.

**Outputs**:
- `gating_network.pt` — PyTorch model state dict
- `gating_weights.mat` — weights exported for MATLAB (W1, b1, W2, b2, W3, b3, X_mean, X_std)

---

## Section 4 — MoE Architecture

### MoE Personalised Fitting — One Patient: `Fit_MoE.m`

Applies the full MoE pipeline to a single real patient selected from
`japan_population_labelled.mat`. Change `PATIENT_IDX` (1–118) to select a different patient.

**Steps:**

1. **Load patient OGTT data** — G_obs [1×5] mmol/L, I_obs [1×5] mU/L, BW (kg)

2. **Gating network forward pass** (pure MATLAB, no Python required):
   ```matlab
   x_norm = ([G_obs, I_obs] - X_mean) ./ X_std;
   h1 = max(0, W1 * x_norm' + b1);
   h2 = max(0, W2 * h1      + b2);
   z  = W3 * h2 + b3;
   w  = double(exp(z - max(z)) / sum(exp(z - max(z))));  % softmax [3x1]
   ```
   Output: `w = [w_NGT, w_IGT, w_T2DM]` — gating weights summing to 1.

3. **lsqnonlin optimisation** of `[k1, k5]`:
   - k6 and k8 remain expert-specific (from the `pids` matrix), mixed via `w`
   - Initial k1: 0.028 (population median); initial k5: `w' * pids(:,1)`
   - Bounds: k1 ∈ [0, 0.05], k5 ∈ [0, 0.17]
   - Error: `[(G_pred - G_obs)/norm(G_obs), (I_pred - I_obs)/norm(I_obs)]`
   - `G_pred = Σ w_e * G_EDES(k1, k5, k6_e, k8_e)` at the 5 sparse time points

4. **Plot** (glucose and insulin, 0–240 min):
   - Dashed coloured curves: three expert EDES simulations
   - Black solid curve: MoE weighted prediction
   - Black circles: observed sparse data

---

### MoE Population Evaluation — Full Dataset: `Dataset_Fit_MoE.m`

Runs the MoE personalised fitting on all N=118 patients in `japan_population_labelled.mat`
and produces three summary figures. Results are saved to `MoE_dataset_results.mat`
(k1_all, k5_all, cats, w_all) for downstream analysis (e.g. `correlation.m`).

**Figure 1 — RMSE per ADA category (boxplot + jittered points)**

For each patient *i*, the MoE model predicts glucose and insulin at the five sparse time
points using the optimised [k1, k5]. RMSE is computed as:
```
G_RMSE(i) = sqrt( mean( (G_pred_sparse(i,:) - G_obs(i,:)).^2 ) )
I_RMSE(i) = sqrt( mean( (I_pred_sparse(i,:) - I_obs(i,:)).^2 ) )
```
Patients are grouped by ADA category. The distribution is shown as a boxplot with
individual patient values overlaid as jittered coloured points.

- A narrow, low box indicates the model fits most patients in that group consistently.
- A wide or high box reveals systematic difficulty for that category.
- Outlier points identify patients where the EDES model cannot reproduce the trajectory.

**Figure 2 — Mean ± 1 SD trajectories per ADA category**

After optimisation, the full MoE trajectory is simulated on a dense grid (0–120 min,
1 min steps). Within each ADA category, the mean and SD of predicted and observed
trajectories are computed. Each subplot shows:
- Shaded band: mean ± 1 SD of observed glucose or insulin
- Coloured line + dots: mean observed values at the five time points
- Black curve: mean of MoE simulated trajectories across patients

- Agreement between the black curve and the coloured line indicates the model captures
  the average population response correctly.
- If the black curve falls inside the shaded band, the model is within one SD of the
  population — a reasonable fit given inter-patient variability.
- A systematic offset reveals a structural limitation of the EDES model for that category.

**Figure 3 — Mean gating weights per ADA category**

For each patient, the gating network produces `[w_NGT, w_IGT, w_T2DM]` (softmax, sum=1).
The mean and SD of each weight within each true ADA category are shown as a grouped bar chart.

- A well-calibrated network assigns the highest weight to the correct expert:
  NGT → high w_NGT, IGT → high w_IGT, T2DM → high w_T2DM.
- Off-diagonal dominance reveals confusion between categories.
- Large error bars indicate the network is uncertain and assigns mixed weights.

---

## Section 5 — Single-Expert Baseline

The single-expert approach uses `Fit_EDES_LatinHyperCube` method as a reference baseline. Unlike MoE, it fits a single EDES model per patient without any population-specific expert mixture or gating network.

**Key differences from MoE:**

| Aspect | MoE | Single-expert |
|--------|-----|---------------|
| Parameters optimised | [k1, k5] | [k1, k5, k6] |
| k6 | Expert-specific, fixed | Free (optimised per patient) |
| k8 | Expert-specific (4.724–7.394) | Fixed at 7.27 |
| Multi-start | No (single init) | Yes — LHS with 5 starts |
| k1 bounds | [0, 0.05] | [0.005, 0.1] |
| k5 bounds | [0, 0.17] | [0, 1] |
| Error function | norm(obs) normalisation | max(obs) + AUC regularisation |
| Optimisation time | 0:1:120 | 0:1:240 |
| Expert mixing | Weighted sum of 3 experts | None |

---

### Single-Expert Fitting — One Patient: `Fit_PID.m`

Fits the single-expert EDES model to one real patient from `japan_population_labelled.mat`.
Change `PATIENT_IDX` (1–118) and `num_par_sets` (number of LHS starts, default 5).

**Steps:**

1. **Load patient OGTT data** and build `input_data` struct compatible with `EDES_ErrorFunc`
2. **LHS multi-start lsqnonlin** on `[k1, k5, k6]`:
   - `num_par_sets` random starting points drawn by Latin Hypercube Sampling
   - Each start runs `EDES_ErrorFunc` (error function with AUC regularisation)
   - Best parameter set selected by minimum resnorm across all starts
3. **Simulate** with `EDES_Parameters` → `EDES_Initial` → `ode45(@EDES_ODE)` on 0–240 min
4. **Plot**: fitted curve (solid) + observed data (circles) for glucose and insulin

Console output shows resnorm per LHS start so convergence behaviour is visible.

---

### Single-Expert Population Evaluation — Full Dataset: `Dataset_Fit_PID.m`

Runs the single-expert fitting on all N=118 Japan dataset patients and produces the same
two summary figures as `Dataset_Fit_MoE.m` (RMSE boxplots and mean ± 1 SD
trajectories) for direct comparison. Results are saved to `single_PID_dataset_results.mat`
(k1_all, k5_all, k6_all, cats).

> **Runtime note**: with 5 LHS starts per patient, this script takes significantly longer
> than `Dataset_Fit_MoE.m`. 

---

## Section 6 — Direct Comparison: `main.m`

Runs both the MoE and single-expert approaches on a single patient and produces
side-by-side comparison figures. Useful for understanding how the two methods
differ in fit quality, trajectory shape, and parameter values for a specific individual.

**Input** (set at the top of the script):
```
PATIENT_IDX   1–118: row index in japan_population_labelled.mat
num_par_sets  number of LHS starts for single-expert (default: 5)
```

To use for a patient not in the Japan dataset, replace the data-loading block with:
```
G_obs      [1 x 5]  plasma glucose at t = [0 30 60 90 120] min  (mmol/L)
I_obs      [1 x 5]  plasma insulin at t = [0 30 60 90 120] min  (mU/L)
BW                  body weight (kg)
POPULATION          'NGT', 'IGT', or 'T2DM'
```

**Figure 1 — Trajectory comparison** (both methods overlaid):
- Blue curve: MoE weighted prediction
- Orange curve: single-expert prediction
- Black circles: observed data
- Subplot titles show all fitted parameter values

**Figure 2 — Performance comparison**:
- Left: grouped bar chart of glucose and insulin RMSE for both methods
  (RMSE computed at the five sparse time points)
- Right: grouped bar chart of k1 and k5 for both methods; k6 (single-expert only)
  is annotated in the title since MoE does not optimise it as a free parameter

---

## Data Files Summary

| File | Location | Format | Created by | Contents |
|------|----------|--------|------------|----------|
| `virtual_population.mat` | `EDES_MoE/Datasets/Virtual Dataset/` | v7.3 | Step 1 | Full virtual population (n × 481 trajectories) |
| `virtual_population_labelled.mat` | `EDES_MoE/Datasets/Virtual Dataset/` | v7.3 | Step 2 | NGT/IGT/T2DM split with ADA labels |
| `virtual_population_sparse.mat` | `EDES_MoE/Datasets/Virtual Dataset/` | v7.3 | Step 3 | 5-point sparse OGTT datasets |
| `pid_predictions.mat` | `EDES_MoE/PID Optimization/` | v7.3 | Step 5 | Expert predictions [N × 3 × 5] |
| `gating_network.pt` | `EDES_MoE/Gating Network/` | PyTorch | Step 6 | Trained model weights |
| `gating_weights.mat` | `EDES_MoE/Gating Network/` | v5 | Step 6 | Weights for MATLAB inference |
| `japan_population_labelled.mat` | `EDES_MoE/Datasets/Real Dataset/` | — | External | 118 real patients: glucose, insulin, BW, GIR |
| `MoE_dataset_results.mat` | `EDES_MoE/Mixture of Experts/` | v5 | `Dataset_Fit_MoE.m` | k1_all, k5_all, cats, w_all (MoE) |
| `single_PID_dataset_results.mat` | `EDES_MoE/Mixture of Experts/` | v5 | `Dataset_Fit_PID.m` | k1_all, k5_all, k6_all, cats (SE) |
| `sample_data.mat` | `EDES_PID/` | v5 | External | Example OGTT data for standalone EDES fitting/testing |

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
  h5py in Python, always call `.T` on the result to get row-major `[n × m]` arrays.
- **float32 vs double**: gating weights exported from PyTorch are float32. In MATLAB,
  always cast the gating output: `w = double(e_z / sum(e_z))` before passing to `lsqnonlin`.
- **param_matrix columns**: `[k1, k5, k6, k8, G_b, I_PL_b, BW]` — column order is
  consistent across all scripts.
- **AUC regularisation constraint**: `EDES_ErrorFunc` accesses `X(1:240,1)` from the ODE
  output to compute the glucose AUC regularisation term. The optimisation time for the
  single-expert approach must therefore span at least 0–239 min (`time_opt = 0:1:240`).
- **The following functions are not modified**: `EDES_ErrorFunc.m`, `EDES_Parameters.m`,
  `EDES_Initial.m`, `Fit_EDES.m`, `Fit_EDES_LatinHyperCube.m` are used as provided.
  The single-expert scripts replicate the LHS logic directly and call `EDES_ErrorFunc`
  with `Display='off'` to suppress verbose output for population-level runs.
- **Path setup**: `startup.m` (root) adds all subfolders to the MATLAB path automatically.
  Run it once (or place it in the MATLAB startup directory) so that scripts in any
  subfolder can call functions from `EDES_PID/` and `EDES_MoE/` without manual `addpath`.
