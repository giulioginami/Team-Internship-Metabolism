# Synthetic Data Improvement (v2) — Japan-Anchored Virtual Population

This branch adds a four-step pipeline that re-grounds the EDES virtual
population on the real **Japan OGTT cohort**, then quantifies the
improvement over the previous generator. The original v1 scripts and
outputs are left untouched so PREVIOUS vs NEW can be compared
side-by-side.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Baseline gap (Japan vs old virtual) | `Quantify_Japan_Virtual_Gap.m` | `japan_virtual_gap_metrics.mat` |
| 2. Characterise the real Japan data    | `Analyze_Japan_Features.m`     | `japan_feature_analysis.mat`   |
| 3a. Re-prior the LHS generator         | `Generate_VirtualPopulation_v2.m` | `virtual_population_v2.mat` |
| 3b. Fixed ADA classifier (OR, not AND) | `Classify_Diabetes_2H_OGTT_v2.m` | (function)                 |
| 3c. Label + balance 33/33/33           | `Label_VirtualPopulation_v2.m`   | `virtual_population_v2_labelled.mat` |
| 4a. Append correlated age / BMI        | `Augment_AgeBMI_v2.m`            | `virtual_population_v2_aug_labelled.mat` |
| 4b. Previous-vs-new comparison figure  | `Compare_Prev_New_Japan.m`       | figure                  |

Run them top-to-bottom; each step only depends on the outputs above it.

## What changed vs v1

- **LHS bounds for `G_b`, `I_PL_b`, `BW`** tightened to Japan p5–p95 ranges
  measured in step 2. Kinetic priors (`k1, k5, k6, k8`) are **left as-is**
  (no Japan ground truth for latent kinetics — decision: *fix bounds only*).
- **T2DM classifier** switched from ADA-AND to ADA-OR, matching the Japan
  study's own labelling convention. The AND rule was rejecting most real
  T2DM patients (median Japan T2DM fasting glucose is 5.78 mmol/L).
- **Class balance** forced to 33/33/33 NGT/IGT/T2DM by stratified
  downsampling, so the training set does not inherit the NGT-heavy
  emergent prior.
- **Age and BMI** appended post-hoc as sampled covariates, drawn from a
  per-category 5-D Gaussian fitted to Japan and conditioned on the already-
  sampled `(G_b, I_PL_b, BW)` — so within-category correlations
  (`age↔G_b ≈ +0.49`, `BMI↔BW ≈ +0.86`) are preserved by construction.
  Neither feeds back into the ODE.

## Headline result

Mean |SMD| across categories / variables / OGTT time points drops from
**1.205 → 0.994** (previous → v2). See `Compare_Prev_New_Japan.m` for the
single-figure summary.

## Known caveats

- The Japan cohort is used both to set the v2 priors and to validate the
  downstream architecture. Architecture metrics on Japan are therefore
  optimistic and must be reported as such.
- LHS sampling stays uniform (decision: minimal/defensible changes).
- A kinetic residual remains and is deferred — not in scope for v2.
