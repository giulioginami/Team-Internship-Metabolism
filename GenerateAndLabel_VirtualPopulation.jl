"""
GenerateAndLabel_VirtualPopulation.jl

Virtual population generation and ADA labelling using the EDES ODE model
and libraries from the juliacon-2024 parameter estimation workshop.

Uses the same model, packages, and conventions as the param estimation scripts:
  - 4-state ODE model (edesode) from juliacon-2024/2_parameter_estimation/
  - StaticArrays for performance
  - QuasiMonteCarlo for Latin Hypercube Sampling
  - OrdinaryDiffEq (Tsit5) for forward simulation

The output is directly compatible with edes_predict_individuals.jl if you want
to run parameter estimation on the virtual population separately.

Pipeline:
  1. Latin Hypercube Sampling of 7 parameters (k1, k5, k6, k8, Gb, Ib, BW)
  2. Forward simulation with the EDES ODE model
  3. Quality control filtering + measurement noise
  4. Clinical feature extraction (12 features)
  5. ADA labelling (NGT / IGT / T2DM)
  6. Output: JLD2 (full data) + CSV (gating network features/labels)

Outputs:
  virtual_population.jld2          - full population data (time series + params)
  gating_network/data/features.csv - 12 clinical features for gating network
  gating_network/data/labels.csv   - integer labels (1=NGT, 2=IGT, 3=T2DM)

Dependencies:
  StaticArrays, OrdinaryDiffEq, QuasiMonteCarlo, Random, Statistics, Printf, JLD2

EDES model reference: Rozendaal et al. (2018)
"""

using StaticArrays
using OrdinaryDiffEq
using QuasiMonteCarlo
using Random
using Statistics
using Printf
using JLD2

# ============================================================================
# SECTION 1 — SETTINGS
# ============================================================================
const N_INDIVIDUALS = 5000             # virtual individuals to attempt
const TIME_SIM      = 0.0:1.0:240.0    # simulation time [min] (matches param estimation)
const MEAL_G        = 75_000.0          # 75 g OGTT in mg
const SEED          = 42               # reproducibility seed

# ============================================================================
# SECTION 2 — PARAMETER BOUNDS FOR LATIN HYPERCUBE SAMPLING
# Order: k1, k5, k6, k8, Gb, Ib, BW
# ============================================================================
const PARAM_NAMES_LHS = ["k1", "k5", "k6", "k8", "Gb", "Ib", "BW"]
const PARAM_LB = [0.005, 0.0,  0.1,  0.5,  3.9,  2.0,  60.0]
const PARAM_UB = [0.04,  0.07, 3.0,  15.0, 12.0, 55.6, 130.0]

# ============================================================================
# SECTION 3 — EDES ODE MODEL
# 4-state ODE from juliacon-2024/2_parameter_estimation/
#
# State vector:  u = [Ggut, Gpl, Ipl, Irem]
#   1  Ggut  glucose mass in gut                 [mg]
#   2  Gpl   plasma glucose concentration        [mmol/L]
#   3  Ipl   plasma insulin concentration        [mU/L]
#   4  Irem  remote/interstitial insulin         [mU/L]
#
# Parameter vector (24 elements):
#   [k1..k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
# ============================================================================
function edesode(u, p, t)
    Ggut, Gpl, Ipl, Irem = u
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
    tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1,
    sigma, Dmeal, bw, Gb, Ib = p

    # gut glucose (Weibull meal appearance)
    dGgut = sigma * k1^sigma * t^(sigma - 1) * exp(-(k1 * t)^sigma) * Dmeal - k2 * Ggut

    # plasma glucose
    gliv  = EGPb - k3 * (Gpl - Gb) - k4 * beta * Irem
    ggut  = k2 * (f / (Vg * bw)) * Ggut
    u_ii  = EGPb * ((Km + Gb) / Gb) * (Gpl / (Km + Gpl))
    u_id  = k5 * beta * Irem * (Gpl / (Km + Gpl))
    u_ren = c1 / (Vg * bw) * (Gpl - Gren) * (Gpl > Gren)
    dGpl  = gliv + ggut - u_ii - u_id - u_ren

    # plasma insulin (PD controller)
    i_pnc = beta^(-1) * (k6 * (Gpl - Gb) + (k7 / tau_i) * Gb + k8 * tau_d * dGpl)
    i_liv = k7 * Gb * Ipl / (beta * tau_i * Ib)
    i_int = k9 * (Ipl - Ib)
    dIpl  = i_pnc - i_liv - i_int
    dIrem = i_int - k10 * Irem

    return SA[dGgut, dGpl, dIpl, dIrem]
end

# ============================================================================
# SECTION 4 — PARAMETER CONSTRUCTION HELPERS
# (follows construct_parameters pattern from param estimation library)
# ============================================================================

"""Build full 24-element SA parameter vector from individual LHS components."""
function build_parameters(k1, k5, k6, k8, Gb, Ib, BW; Dmeal=MEAL_G)
    SA[k1, 0.28, 6.07e-3, 2.35e-4, k5, k6, 1.15, k8,
       3.83e-2, 2.84e-1, 31.0, 3.0, 1.0, 9.0, 0.043, 13.2,
       0.005551, 17.0/70.0, 0.1, 1.4, Dmeal, BW, Gb, Ib]
end

# ============================================================================
# SECTION 5 — FORWARD SIMULATION
# ============================================================================

"""
    run_edes(params, tgrid) -> (G_sim, I_sim) or nothing

Run a single EDES ODE simulation. Returns glucose and insulin vectors
on tgrid, or nothing on solver failure.
"""
function run_edes(params, tgrid)
    Gb = params[23]
    Ib = params[24]
    u0    = SA[0.0, Gb, Ib, 0.0]
    tspan = (Float64(first(tgrid)), Float64(last(tgrid)))

    prob = ODEProblem(edesode, u0, tspan, params)
    sol  = try
        solve(prob, Tsit5(); saveat=collect(tgrid), reltol=1e-5, abstol=1e-8)
    catch
        return nothing
    end

    if string(sol.retcode) ∉ ("Success", "Default") || length(sol.t) != length(tgrid)
        return nothing
    end

    G_sim = [sol.u[i][2] for i in eachindex(sol.t)]   # Gpl
    I_sim = [sol.u[i][3] for i in eachindex(sol.t)]   # Ipl
    return G_sim, I_sim
end

# ============================================================================
# SECTION 6 — FEATURE EXTRACTION
# (same 12 clinical features as gating_network/mock_data_generation.jl)
# ============================================================================
const FEATURE_NAMES = [
    "fasting_glucose", "peak_glucose", "time_peak_glucose",
    "glucose_120", "glucose_auc",
    "fasting_insulin", "peak_insulin", "time_peak_insulin",
    "insulin_30", "insulin_120", "insulin_auc",
    "homa_ir"
]

"""Extract 12 clinically motivated features from glucose/insulin time series."""
function extract_features(glucose, insulin, time_vec)
    at(signal, minute) = begin
        idx = findfirst(x -> x >= minute, time_vec)
        idx === nothing ? signal[end] : signal[idx]
    end

    fasting_glucose   = at(glucose, 0)
    peak_glucose      = maximum(glucose)
    time_peak_glucose = Float64(time_vec[argmax(glucose)])
    glucose_120       = at(glucose, 120)
    glucose_auc       = sum((glucose[1:end-1] .+ glucose[2:end]) ./ 2 .* diff(time_vec))

    fasting_insulin   = at(insulin, 0)
    peak_insulin      = maximum(insulin)
    time_peak_insulin = Float64(time_vec[argmax(insulin)])
    insulin_30        = at(insulin, 30)
    insulin_120       = at(insulin, 120)
    insulin_auc       = sum((insulin[1:end-1] .+ insulin[2:end]) ./ 2 .* diff(time_vec))

    homa_ir = fasting_glucose * fasting_insulin / 22.5

    return [
        fasting_glucose, peak_glucose, time_peak_glucose,
        glucose_120, glucose_auc,
        fasting_insulin, peak_insulin, time_peak_insulin,
        insulin_30, insulin_120, insulin_auc,
        homa_ir
    ]
end

# ============================================================================
# SECTION 7 — ADA LABELLING
# ADA criteria (applied to glucose_noisy):
#   T2DM : fasting >= 7.0  AND  2h >= 11.1
#   IGT  : NOT T2DM  AND  7.8 <= 2h <= 11.1
#   NGT  : fasting < 5.6  AND  2h < 7.8
# ============================================================================

"""
    label_population(glucose_noisy, time_vec)

Apply ADA diagnostic criteria. Returns string labels, integer labels,
fasting glucose, 2h glucose, and boolean masks for each category.
"""
function label_population(glucose_noisy::Matrix, time_vec)
    idx_fast = findfirst(==(0.0),   time_vec)
    idx_2h   = findfirst(==(120.0), time_vec)
    (idx_fast === nothing || idx_2h === nothing) &&
        error("Cannot locate t=0 or t=120 in the time vector.")

    G_fasting = glucose_noisy[:, idx_fast]
    G_2h      = glucose_noisy[:, idx_2h]

    is_T2DM = (G_fasting .>= 7.0) .& (G_2h .>= 11.1)
    is_IGT  = .!is_T2DM .& (G_2h .>= 7.8) .& (G_2h .<= 11.1)
    is_NGT  = (G_fasting .< 5.6)  .& (G_2h .< 7.8)

    labels_str = fill("NGT", size(glucose_noisy, 1))
    labels_str[is_IGT]  .= "IGT"
    labels_str[is_T2DM] .= "T2DM"

    # Integer labels for gating network: 1=NGT(Healthy), 2=IGT, 3=T2DM
    labels_int = ones(Int, size(glucose_noisy, 1))
    labels_int[is_IGT]  .= 2
    labels_int[is_T2DM] .= 3

    return labels_str, labels_int, G_fasting, G_2h, is_NGT, is_IGT, is_T2DM
end

# ============================================================================
# SECTION 8 — DATA I/O
# ============================================================================

"""Save features and labels as CSV for the gating network."""
function save_gating_data(output_dir, features, labels_int)
    mkpath(output_dir)

    # features.csv
    open(joinpath(output_dir, "features.csv"), "w") do io
        println(io, join(FEATURE_NAMES, ","))
        for i in axes(features, 1)
            println(io, join(features[i, :], ","))
        end
    end

    # labels.csv
    open(joinpath(output_dir, "labels.csv"), "w") do io
        println(io, "label")
        for l in labels_int
            println(io, l)
        end
    end

    @info "Saved gating network data to $output_dir ($(size(features, 1)) patients)"
end

# ============================================================================
# SECTION 9 — VIRTUAL POPULATION GENERATION
# ============================================================================

function generate_virtual_population(;
        N_sim::Int      = N_INDIVIDUALS,
        seed::Int       = SEED,
        tgrid           = TIME_SIM,
        meal_g::Float64 = MEAL_G)

    Random.seed!(seed)
    n_t      = length(tgrid)
    time_vec = collect(Float64, tgrid)

    # ── Latin Hypercube Sampling (QuasiMonteCarlo) ────────────────────────
    println("Generating LHS design ($N_sim x 7) via QuasiMonteCarlo...")
    lhs = QuasiMonteCarlo.sample(N_sim, PARAM_LB, PARAM_UB, LatinHypercubeSample())
    lhs = Matrix(lhs')   # 7 x N_sim → N_sim x 7

    # ── Pre-allocate (NaN = not yet accepted) ─────────────────────────────
    glucose_clean = fill(NaN, N_sim, n_t)
    insulin_clean = fill(NaN, N_sim, n_t)
    glucose_noisy = fill(NaN, N_sim, n_t)
    insulin_noisy = fill(NaN, N_sim, n_t)
    param_matrix  = fill(NaN, N_sim, 7)
    valid_flag    = falses(N_sim)

    println("Starting virtual population generation (N=$N_sim)...")

    for i in 1:N_sim
        if mod(i, 100) == 0
            n_valid = sum(valid_flag)
            println("  Individual $i/$N_sim | Valid so far: $n_valid")
        end

        # ── Unpack LHS samples ──────────────────────────────────────────
        k1 = lhs[i, 1]; k5 = lhs[i, 2]; k6 = lhs[i, 3]
        k8 = lhs[i, 4]; Gb = lhs[i, 5]; Ib = lhs[i, 6]
        BW = lhs[i, 7]

        # ── Build full parameter vector (SA) ─────────────────────────────
        params = build_parameters(k1, k5, k6, k8, Gb, Ib, BW; Dmeal=meal_g)

        # ── Forward simulation ───────────────────────────────────────────
        result = run_edes(params, tgrid)
        result === nothing && continue

        G_sim, I_sim = result

        # ── Quality control filters ──────────────────────────────────────
        (any(G_sim .< 0.0) || any(I_sim .< 0.0)) && continue
        (maximum(G_sim) > 30.0 || minimum(G_sim) < 2.0) && continue
        (maximum(I_sim) > 200.0 || minimum(I_sim) < 0.0) && continue

        # ── Measurement noise ────────────────────────────────────────────
        noise_G_pct = 0.02 + 0.01 * rand()        # 2-3 % CV
        noise_I_pct = 0.05 + 0.03 * rand()        # 5-8 % CV
        G_noisy = G_sim .* (1.0 .+ noise_G_pct .* randn(n_t))
        I_noisy = I_sim .* (1.0 .+ noise_I_pct .* randn(n_t))
        G_noisy = max.(G_noisy, 0.0)
        I_noisy = max.(I_noisy, 0.0)

        # ── Store accepted individual ────────────────────────────────────
        glucose_clean[i, :] = G_sim
        insulin_clean[i, :] = I_sim
        glucose_noisy[i, :] = G_noisy
        insulin_noisy[i, :] = I_noisy
        param_matrix[i, :]  = [k1, k5, k6, k8, Gb, Ib, BW]
        valid_flag[i]       = true
    end

    # ── Trim to accepted individuals ──────────────────────────────────────
    idx_valid = findall(valid_flag)
    n_valid   = length(idx_valid)
    @printf("\nSimulation complete. Accepted: %d / %d individuals (%.1f%%)\n",
            n_valid, N_sim, 100.0 * n_valid / N_sim)

    glucose_clean = glucose_clean[idx_valid, :]
    insulin_clean = insulin_clean[idx_valid, :]
    glucose_noisy = glucose_noisy[idx_valid, :]
    insulin_noisy = insulin_noisy[idx_valid, :]
    param_matrix  = param_matrix[idx_valid, :]

    # ── Feature extraction ────────────────────────────────────────────────
    println("\nExtracting 12 clinical features...")
    features = zeros(n_valid, 12)
    for i in 1:n_valid
        features[i, :] = extract_features(
            glucose_noisy[i, :], insulin_noisy[i, :], time_vec)
    end

    # ── ADA labelling ─────────────────────────────────────────────────────
    labels_str, labels_int, G_fasting, G_2h, is_NGT, is_IGT, is_T2DM =
        label_population(glucose_noisy, time_vec)

    n_NGT  = sum(is_NGT)
    n_IGT  = sum(is_IGT)
    n_T2DM = sum(is_T2DM)
    @printf("\nADA labelling results (N = %d):\n", n_valid)
    @printf("  NGT  : %4d  (%.1f%%)\n", n_NGT,  100.0 * n_NGT  / n_valid)
    @printf("  IGT  : %4d  (%.1f%%)\n", n_IGT,  100.0 * n_IGT  / n_valid)
    @printf("  T2DM : %4d  (%.1f%%)\n", n_T2DM, 100.0 * n_T2DM / n_valid)

    # ── Build output NamedTuple ───────────────────────────────────────────
    vp = (
        time          = time_vec,
        glucose_clean = glucose_clean,
        insulin_clean = insulin_clean,
        glucose_noisy = glucose_noisy,
        insulin_noisy = insulin_noisy,
        param_matrix  = param_matrix,
        param_names   = PARAM_NAMES_LHS,
        features      = features,
        feature_names = FEATURE_NAMES,
        labels        = labels_str,
        labels_int    = labels_int,
        G_fasting     = G_fasting,
        G_2h          = G_2h,
        is_NGT        = is_NGT,
        is_IGT        = is_IGT,
        is_T2DM       = is_T2DM,
        n_valid       = n_valid,
        N_attempted   = N_sim,
    )

    return vp
end

# ============================================================================
# SECTION 10 — ENTRY POINT
# ============================================================================

function main()
    # ── Step 1: Generate virtual population ──────────────────────────────
    vp = generate_virtual_population()

    # ── Step 2: Save JLD2 (full data) ────────────────────────────────────
    outfile = joinpath(@__DIR__, "virtual_population.jld2")
    println("\nSaving $outfile...")
    jldsave(outfile; virtual_population=vp)
    println("  Saved: $outfile")

    # ── Step 3: Save CSV for gating network ──────────────────────────────
    gating_dir = joinpath(@__DIR__, "gating_network", "data")
    save_gating_data(gating_dir, vp.features, vp.labels_int)

    println("\nPipeline complete.")
    println("  JLD2 : $outfile")
    println("  CSV  : $gating_dir/features.csv, labels.csv")
    println("\nNext: run gating_network/gating_network.jl to train the classifier")
    println("  For parameter estimation, use edes_predict_individuals.jl on the JLD2 data")

    return vp
end

# ── Run automatically when called as a script ─────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
