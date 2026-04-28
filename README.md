# Mixture of Experts for Personalised Glucose-Insulin Modelling

A Mixture of Experts (MoE) framework that classifies patients into metabolic types and routes them to specialised models of glucose-insulin dynamics, built on the **Eindhoven Diabetes Education Simulator (EDES)**.

## Background

### The EDES Model

The [EDES model](https://pubmed.ncbi.nlm.nih.gov/25526760/) (Maas et al., 2015) is a mechanistic ODE model describing plasma glucose and insulin concentrations after a meal. It captures:

- **Gut glucose absorption** via a Weibull appearance function
- **Plasma glucose regulation** — hepatic production, insulin-dependent uptake, renal clearance
- **Insulin secretion and clearance** — a PID-style controller responding to glucose levels

The model uses 4 state variables and 24 parameters. Three parameters are particularly patient-specific:

| Parameter | Role |
|-----------|------|
| `k1` | Rate of gastric emptying |
| `k5` | Peripheral insulin sensitivity |
| `k6` | Pancreatic insulin secretion gain |

These parameters vary across patient types: healthy individuals have high insulin sensitivity (`k5`) and normal secretion (`k6`), while T2D patients have low sensitivity and impaired secretion.

### Why Mixture of Experts?

A single model cannot capture the full spectrum of metabolic responses across healthy, pre-diabetic, and diabetic individuals. The MoE approach:

1. **Classifies** each patient into a metabolic type using a gating network
2. **Routes** them to a specialised expert model tuned for that type
3. **Combines** expert predictions weighted by classification confidence

This allows each expert to specialise in a narrower physiological range, improving personalised predictions.

### ADA Diagnostic Criteria

Patient classification follows the [American Diabetes Association](https://diabetes.org) diagnostic thresholds applied to a 75g oral glucose tolerance test (OGTT):

| Diagnosis | Fasting Glucose | 2-hour Glucose |
|-----------|----------------|----------------|
| **NGT** (Normal Glucose Tolerance) | < 5.6 mmol/L | < 7.8 mmol/L |
| **IGT** (Impaired Glucose Tolerance) | -- | 7.8 -- 11.1 mmol/L |
| **T2DM** (Type 2 Diabetes Mellitus) | >= 7.0 mmol/L | >= 11.1 mmol/L |

## Architecture

```
                  ┌──────────────────────────────────────────────────────────┐
                  │             Virtual Population Generator                 │
                  │       (GenerateAndLabel_VirtualPopulation.jl)            │
                  │                                                          │
                  │  LHS (5000 patients) -> ODE simulation -> 12 features   │
                  │  + ADA labelling -> features.csv + labels.csv           │
                  └──────────────────────────┬───────────────────────────────┘
                                             │
                                             ▼
┌──────────┐     ┌────────────────┐     ┌─────────────────┐
│ Features │ ──> │ Gating Network │ ──> │ w1, w2, w3      │
│ (12-dim) │     │ (Flux.jl NN)   │     │ expert weights   │
└──────────┘     └────────────────┘     └────────┬────────┘
                                                  │
                    ┌─────────────────────────────┼──────────────────────────┐
                    ▼                             ▼                          ▼
              ┌───────────┐                ┌───────────┐               ┌───────────┐
              │ Expert 1  │                │ Expert 2  │               │ Expert 3  │
              │ (Healthy) │                │  (IGT)    │               │  (T2D)    │
              └─────┬─────┘                └─────┬─────┘               └─────┬─────┘
                    │                            │                           │
                    └────────────────────────────┼───────────────────────────┘
                                                 ▼
                                       y_hat = Sum( wi * expert_i(x) )
```

## Repository Structure

```
team_internship/
├── GenerateAndLabel_VirtualPopulation.jl   # Virtual population generation pipeline
├── gating_network/
│   ├── gating_network.jl                   # Gating network (Flux.jl classifier)
│   ├── mock_data_generation.jl             # Legacy placeholder data generator
│   └── data/
│       ├── features.csv                    # 12 clinical features (generated)
│       └── labels.csv                      # ADA labels: 1=NGT, 2=IGT, 3=T2DM
├── virtual_population.jld2                 # Full population data (JLD2 binary)
├── juliacon-2024/                          # JuliaCon 2024 workshop (reference material)
│   ├── 1_implementation/                   # EDES model implementations (ODE + DDE)
│   ├── 2_parameter_estimation/             # Parameter estimation with PREDICT data
│   ├── 3_identifiability/                  # Profile likelihood analysis
│   └── 4_scientific_ml/                    # Hybrid neural ODE approach
├── EDES_noGint/                            # Original MATLAB implementation
├── edes_frontend.py                        # Interactive Dash/Plotly web frontend
└── *.pdf                                   # Reference papers
```

## Pipeline

### Step 1 — Generate Virtual Population

`GenerateAndLabel_VirtualPopulation.jl` creates a synthetic cohort of patients:

```
julia GenerateAndLabel_VirtualPopulation.jl
```

**What it does:**

1. **Latin Hypercube Sampling** — Samples 5000 patients across 7 parameters (`k1`, `k5`, `k6`, `k8`, `Gb`, `Ib`, `BW`) using [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl), ensuring uniform coverage of the parameter space
2. **Forward ODE simulation** — Runs the 4-state EDES model ([OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), Tsit5 solver) for each patient over a 240-minute OGTT
3. **Quality control** — Filters out physiologically implausible results (negative values, extreme glucose > 30 mmol/L, etc.). Acceptance rate: ~79%
4. **Measurement noise** — Adds realistic assay noise (2-3% CV for glucose, 5-8% CV for insulin)
5. **Feature extraction** — Computes 12 clinically motivated features per patient (see below)
6. **ADA labelling** — Classifies each patient as NGT, IGT, or T2DM based on fasting and 2-hour glucose

**LHS Parameter Bounds:**

| Parameter | Lower | Upper | Description |
|-----------|-------|-------|-------------|
| `k1` | 0.005 | 0.04 | Gastric emptying rate |
| `k5` | 0.0 | 0.07 | Insulin-dependent glucose uptake |
| `k6` | 0.1 | 3.0 | Pancreatic insulin secretion gain |
| `k8` | 0.5 | 15.0 | Derivative gain of insulin secretion |
| `Gb` | 3.9 | 12.0 | Fasting glucose (mmol/L) |
| `Ib` | 2.0 | 55.6 | Fasting insulin (mU/L) |
| `BW` | 60.0 | 130.0 | Body weight (kg) |

**Outputs:**

| File | Content |
|------|---------|
| `virtual_population.jld2` | Full population data: time series, parameters, features, labels |
| `gating_network/data/features.csv` | 12 features x ~3955 patients |
| `gating_network/data/labels.csv` | Integer labels (1=NGT, 2=IGT, 3=T2DM) |

### Step 2 — Train the Gating Network

```
julia gating_network/gating_network.jl
```

**What it does:**

Trains a feedforward neural network ([Flux.jl](https://fluxml.ai/)) that classifies patients from the 12 clinical features into the 3 metabolic types.

**12 Clinical Features (input):**

| # | Feature | Unit | Description |
|---|---------|------|-------------|
| 1 | `fasting_glucose` | mmol/L | Glucose at t=0 |
| 2 | `peak_glucose` | mmol/L | Maximum glucose during OGTT |
| 3 | `time_peak_glucose` | min | Time to peak glucose |
| 4 | `glucose_120` | mmol/L | Glucose at t=120 min (2-hour value) |
| 5 | `glucose_auc` | mmol/L * min | Area under glucose curve |
| 6 | `fasting_insulin` | mU/L | Insulin at t=0 |
| 7 | `peak_insulin` | mU/L | Maximum insulin during OGTT |
| 8 | `time_peak_insulin` | min | Time to peak insulin |
| 9 | `insulin_30` | mU/L | Insulin at t=30 min (early-phase secretion) |
| 10 | `insulin_120` | mU/L | Insulin at t=120 min |
| 11 | `insulin_auc` | mU/L * min | Area under insulin curve |
| 12 | `homa_ir` | -- | HOMA-IR index (fasting_glucose * fasting_insulin / 22.5) |

**Network architecture:**

```
Input (12) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(3) -> Softmax
```

No dropout is used because the labels are a **deterministic function** of features 1 and 4 (fasting glucose and 2-hour glucose), so regularisation only hurts convergence.

**Training details:**

| Setting | Value |
|---------|-------|
| Optimiser | Adam (lr = 5e-4) |
| Loss | Weighted cross-entropy (inverse-frequency class weights) |
| Batch size | 32 |
| Max epochs | 1000 |
| Early stopping | Patience = 100 epochs, or perfect accuracy |
| Train/test split | 80/20 stratified |
| Normalisation | Z-score (mean/std from training set) |

**Why class-weighted loss?** Broad LHS sampling produces many more NGT patients than IGT or T2DM. Without weighting, the network can achieve ~80% accuracy by predicting "healthy" for everyone. Inverse-frequency weights force it to learn the minority class boundaries.

**Target accuracy: 100%** — Since ADA labels are a deterministic function of fasting glucose and glucose at 120 min (both included as features), perfect classification is theoretically achievable and is the target.

**Output:**

The trained model is wrapped in a `GatingPredictor` struct that can be used directly in the MoE pipeline:

```julia
predictor, history = train_and_build(joinpath(@__DIR__, "data"))

# Single patient: returns [P(healthy), P(IGT), P(T2D)]
weights = predict_gates(predictor, patient_features)

# Batch: returns 3 x n_patients matrix
weights = predict_gates_batch(predictor, features_matrix)

# Hard classification: returns 1, 2, or 3
label = classify(predictor, patient_features)
```

## What We Started From

This project builds on the **JuliaCon 2024 workshop** on personalised glucose-insulin modelling, which provided:

- The **EDES ODE model** implementation using StaticArrays and OrdinaryDiffEq
- A **parameter estimation** pipeline using Optimization.jl with the PREDICT clinical dataset
- **Profile likelihood** analysis for parameter identifiability
- A **hybrid neural ODE** (SciML) approach for learning glucose appearance rates

The workshop code is in `juliacon-2024/` and serves as the foundation for the libraries and conventions used throughout this project.

### Evolution of the codebase

1. **Starting point** — `juliacon-2024/` workshop code: EDES model implementation, parameter estimation with PREDICT data, identifiability analysis
2. **Mock data generator** — `gating_network/mock_data_generation.jl`: a DDE-based placeholder that simulated patients using hardcoded parameter centres per type with log-normal noise. Used for initial gating network development.
3. **Virtual population generator** — `GenerateAndLabel_VirtualPopulation.jl` (by Giulio): originally used custom LHS and the 5-state DDE model. Refactored to use the same 4-state ODE model and libraries as the parameter estimation workshop (StaticArrays, OrdinaryDiffEq, QuasiMonteCarlo).
4. **Gating network improvements** — upgraded from a small network with dropout (`[32, 16]` + Dropout) to a larger architecture (`[128, 64, 32]`, no dropout) with class-weighted loss, targeting 100% accuracy on the deterministic ADA labels.

## What We Achieved

- A **complete end-to-end pipeline** from virtual patient generation to trained classifier
- **Consistent use of the EDES 4-state ODE model** across all components (simulation, feature extraction, classification)
- A **virtual population of ~3955 patients** spanning healthy, pre-diabetic, and diabetic physiology, generated via Latin Hypercube Sampling
- **12 clinically meaningful features** extracted from simulated OGTT responses
- **ADA-compliant diagnostic labelling** (NGT / IGT / T2DM)
- A **gating network optimised for 100% accuracy** on deterministic labels, with class-weighted loss to handle population imbalance
- A **portable `GatingPredictor` struct** ready for integration with downstream expert models

## Dependencies

### Julia packages

```julia
using Pkg
Pkg.add(["StaticArrays", "OrdinaryDiffEq", "QuasiMonteCarlo", "JLD2", "Flux"])
```

| Package | Used by | Purpose |
|---------|---------|---------|
| StaticArrays | Virtual pop. generator | Fast parameter vectors (SVector) |
| OrdinaryDiffEq | Virtual pop. generator | ODE solver (Tsit5) |
| QuasiMonteCarlo | Virtual pop. generator | Latin Hypercube Sampling |
| JLD2 | Virtual pop. generator | Binary data serialisation |
| Flux | Gating network | Neural network training |

### Python (optional, for frontend only)

```
pip install dash plotly numpy scipy
```

## Quick Start

```bash
# 1. Install Julia packages (first time only)
julia -e 'using Pkg; Pkg.add(["StaticArrays", "OrdinaryDiffEq", "QuasiMonteCarlo", "JLD2", "Flux"])'

# 2. Generate virtual population (~3955 patients)
julia GenerateAndLabel_VirtualPopulation.jl

# 3. Train the gating network
julia gating_network/gating_network.jl
```

## References

- Maas, A.H. et al. (2015). *A physiology-based model describing heterogeneity in glucose metabolism.* [PubMed](https://pubmed.ncbi.nlm.nih.gov/25526760/)
- O'Donovan, S. et al. (2022). *Extending the EDES model with triglycerides and NEFA.* [iScience](https://www.cell.com/iscience/fulltext/S2589-0042(22)01478-X)
- Berry, S.E. et al. (2020). *Human postprandial responses to food and potential for precision nutrition.* [Nature Medicine](https://doi.org/10.1038/s41591-020-0934-0)
- American Diabetes Association. *Standards of Care in Diabetes.* [diabetes.org](https://diabetes.org)

## Team

Built as part of the Q3 team internship at [Eindhoven University of Technology](https://www.tue.nl/), Department of Biomedical Engineering.

Based on the JuliaCon 2024 workshop by [Shauna O'Donovan](https://research.tue.nl/en/persons/shauna-odonovan), [Max de Rooij](https://research.tue.nl/en/persons/max-de-rooij), and [Natal van Riel](https://research.tue.nl/en/persons/natal-aw-van-riel).
