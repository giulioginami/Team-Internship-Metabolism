#=
    Mock Data Generation for MoE Gating Network
    =============================================
    Simulates patient meal response data using the EDES DDE model for 3 patient types:
      1 = Healthy (normal glucose tolerance)
      2 = IGT (impaired glucose tolerance)
      3 = T2D (type 2 diabetes)

    This script is a PLACEHOLDER — it will be replaced by the actual Julia data
    generation script being developed by a colleague. The interface contract is:
      - Output: data/features.csv  (n_patients × 12 feature columns + header)
      - Output: data/labels.csv    (n_patients × 1 label column + header)

    Usage:
      julia mock_data_generation.jl            # generates 100 patients per type
      julia mock_data_generation.jl 200        # generates 200 patients per type
=#

using StaticArrays
using DelayDiffEq
using Random
using Statistics

# ─── EDES DDE Model ──────────────────────────────────────────────────────────
# Replicates juliacon-2024/1_implementation/edes_dde.jl

function configure_history(Gb)
    h(p, t; idxs=nothing) = typeof(idxs) <: Number ? Gb : ones(5) .* Gb
    return h
end

function edesdde(u, h, p, t)
    Ggut, Gpl, Gint, Ipl, Irem = u
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10,
    tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1,
    t_int, sigma, Dmeal, bw, Gb, Ib = p

    Ghist = h(p, t - t_int; idxs=2)

    # gut glucose
    dGgut = sigma * k1^sigma * t^(sigma - 1) * exp(-(k1 * t)^sigma) * Dmeal - k2 * Ggut

    # plasma glucose
    gliv  = EGPb - k3 * (Gpl - Gb) - k4 * beta * Irem
    ggut  = k2 * (f / (Vg * bw)) * Ggut
    u_ii  = EGPb * ((Km + Gb) / Gb) * (Gpl / (Km + Gpl))
    u_id  = k5 * beta * Irem * (Gpl / (Km + Gpl))
    u_ren = c1 / (Vg * bw) * (Gpl - Gren) * (Gpl > Gren)

    dGpl  = gliv + ggut - u_ii - u_id - u_ren
    dGint = Gpl - Ghist

    # plasma insulin
    i_pnc = beta^(-1) * (k6 * (Gpl - Gb) + (k7 / tau_i) * (Gint + Gb) + k8 * tau_d * dGpl)
    i_liv = k7 * Gb * Ipl / (beta * tau_i * Ib)
    i_int = k9 * (Ipl - Ib)
    dIpl  = i_pnc - i_liv - i_int
    dIrem = i_int - k10 * Irem

    return SA[dGgut, dGpl, dGint, dIpl, dIrem]
end

# ─── Patient Type Definitions ────────────────────────────────────────────────
# Parameters shared across all patient types (fixed physiology / meal protocol)

const SHARED = (
    k2=0.28, k3=6.07e-3, k4=2.35e-4, k7=1.15, k8=7.27,
    k9=3.83e-2, k10=2.84e-1, tau_i=31.0, tau_d=3.0,
    Gren=9.0, EGPb=0.043, Km=13.2, f=0.005551,
    Vg=17.0 / 70.0, c1=0.1, t_int=30.0, sigma=1.4,
    Dmeal=75.0e3, bw=70.0
)

# Type-specific parameter centres (k1, k5, k6, beta, Gb, Ib).
# k5 = peripheral insulin sensitivity, k6 = proportional gain of PID insulin
# secretion, Gb/Ib = fasting glucose/insulin.
# Presets aligned with edes_frontend.py + clinical physiology.
const TYPE_CENTRES = Dict(
    :healthy => (k1=0.0105, k5=0.0424,       k6=2.2975,       beta=1.0, Gb=5.0, Ib=10.0),
    :igt     => (k1=0.0095, k5=0.0424 * 0.4, k6=2.2975 * 1.3, beta=1.0, Gb=5.8, Ib=14.0),
    :t2d     => (k1=0.0085, k5=0.0424 * 0.1, k6=2.2975 * 0.4, beta=1.0, Gb=7.5, Ib=18.0),
)

# Coefficient of variation for inter-patient variability (log-normal noise).
const PARAM_CV = (k1=0.10, k5=0.15, k6=0.12, Gb=0.08, Ib=0.12, bw=0.15)

# ─── Simulation helpers ──────────────────────────────────────────────────────

"""Sample one patient's full parameter vector with stochastic variability."""
function sample_patient_params(ptype::Symbol, rng::AbstractRNG)
    tc = TYPE_CENTRES[ptype]
    vary(val, cv) = val * exp(cv * randn(rng))

    k1 = vary(tc.k1, PARAM_CV.k1)
    k5 = max(vary(tc.k5, PARAM_CV.k5), 1e-6)   # keep positive
    k6 = max(vary(tc.k6, PARAM_CV.k6), 0.05)
    Gb = vary(tc.Gb, PARAM_CV.Gb)
    Ib = vary(tc.Ib, PARAM_CV.Ib)
    bw = vary(SHARED.bw, PARAM_CV.bw)

    return SA[
        k1, SHARED.k2, SHARED.k3, SHARED.k4, k5, k6, SHARED.k7, SHARED.k8,
        SHARED.k9, SHARED.k10, SHARED.tau_i, SHARED.tau_d, tc.beta,
        SHARED.Gren, SHARED.EGPb, SHARED.Km, SHARED.f, SHARED.Vg, SHARED.c1,
        SHARED.t_int, SHARED.sigma, SHARED.Dmeal, bw, Gb, Ib
    ]
end

"""Run the EDES DDE for one patient, saving every 1 min from 0–240 min."""
function simulate_patient(params)
    Gb = params[24]
    Ib = params[25]
    u0 = SA[0.0, Gb, Gb, Ib, 0.0]
    h  = configure_history(Gb)
    prob = DDEProblem(edesdde, u0, h, (0.0, 240.0), params)
    return solve(prob, MethodOfSteps(Tsit5()); saveat=1.0, abstol=1e-8, reltol=1e-8)
end

# ─── Feature Extraction ─────────────────────────────────────────────────────

const FEATURE_NAMES = [
    "fasting_glucose", "peak_glucose", "time_peak_glucose",
    "glucose_120", "glucose_auc",
    "fasting_insulin", "peak_insulin", "time_peak_insulin",
    "insulin_30", "insulin_120", "insulin_auc",
    "homa_ir"
]

# Real-world OGTT sampling schedule, matched to juliacon-2024/predict/*.csv
# so the model trains and evaluates on the same time grid.
const GLUCOSE_TIMES = (0, 15, 30, 60, 120, 180, 240)
const INSULIN_TIMES = (0, 15, 30, 60, 120, 240)

"""
    extract_sparse_measurements(sol; glucose_noise, insulin_noise, rng)
        -> (glucose::Vector, insulin::Vector)

Subsample the dense simulation at the OGTT schedule and add measurement noise.
Used to train the gating network on raw sparse measurements (the way the model
will see real-life patient data) instead of derived features.
"""
function extract_sparse_measurements(sol;
        glucose_noise=0.05, insulin_noise=0.15,
        rng=Random.default_rng())
    t       = sol.t
    glucose = sol[2, :]
    insulin = sol[4, :]

    # noise applied independently from extract_features so the two signals
    # remain consistent when we save both for the same patient.
    glucose = glucose .+ glucose .* glucose_noise .* randn(rng, length(glucose))
    insulin = insulin .+ insulin .* insulin_noise .* randn(rng, length(insulin))
    glucose = max.(glucose, 0.5)
    insulin = max.(insulin, 0.0)

    at(signal, minute) = signal[clamp(round(Int, minute) + 1, 1, length(signal))]

    g = Float64[at(glucose, m) for m in GLUCOSE_TIMES]
    i = Float64[at(insulin, m) for m in INSULIN_TIMES]
    return g, i
end

"""
    extract_features(sol; glucose_noise, insulin_noise, rng) -> Vector{Float64}

Extract 12 clinically motivated features from a meal response simulation.
Measurement noise is added to mimic real assay variability.
"""
function extract_features(sol;
        glucose_noise=0.05, insulin_noise=0.15,
        rng=Random.default_rng())
    t = sol.t                 # 0,1,2,...,240
    glucose = sol[2, :]       # Gpl  (mmol/L)
    insulin = sol[4, :]       # Ipl  (mU/L)

    # add measurement noise
    glucose = glucose .+ glucose .* glucose_noise .* randn(rng, length(glucose))
    insulin = insulin .+ insulin .* insulin_noise .* randn(rng, length(insulin))
    glucose = max.(glucose, 0.5)
    insulin = max.(insulin, 0.0)

    # helper: value at nearest minute
    at(signal, minute) = signal[clamp(round(Int, minute) + 1, 1, length(signal))]

    # glucose features
    fasting_glucose   = at(glucose, 0)
    peak_glucose      = maximum(glucose)
    time_peak_glucose = Float64(t[argmax(glucose)])
    glucose_120       = at(glucose, 120)
    glucose_auc       = sum((glucose[1:end-1] .+ glucose[2:end]) ./ 2 .* diff(t))

    # insulin features
    fasting_insulin   = at(insulin, 0)
    peak_insulin      = maximum(insulin)
    time_peak_insulin = Float64(t[argmax(insulin)])
    insulin_30        = at(insulin, 30)
    insulin_120       = at(insulin, 120)
    insulin_auc       = sum((insulin[1:end-1] .+ insulin[2:end]) ./ 2 .* diff(t))

    # derived: HOMA-IR  (fasting glucose [mmol/L] * fasting insulin [mU/L] / 22.5)
    homa_ir = fasting_glucose * fasting_insulin / 22.5

    return [
        fasting_glucose, peak_glucose, time_peak_glucose,
        glucose_120, glucose_auc,
        fasting_insulin, peak_insulin, time_peak_insulin,
        insulin_30, insulin_120, insulin_auc,
        homa_ir
    ]
end

# ─── Cohort Generation ───────────────────────────────────────────────────────

"""
    generate_cohort(n_per_type; seed) -> (features::Matrix, labels::Vector{Int})

Generate a balanced cohort.  Returns (n_total × 12) feature matrix and
integer labels (1=healthy, 2=IGT, 3=T2D).
"""
function generate_cohort(n_per_type::Int; seed=42)
    rng = MersenneTwister(seed)
    types     = [:healthy, :igt, :t2d]
    label_map = Dict(:healthy => 1, :igt => 2, :t2d => 3)

    all_features = Vector{Vector{Float64}}()
    all_glucose  = Vector{Vector{Float64}}()
    all_insulin  = Vector{Vector{Float64}}()
    all_labels   = Vector{Int}()

    for ptype in types
        generated = 0
        attempts  = 0
        while generated < n_per_type && attempts < n_per_type * 5
            attempts += 1
            try
                params = sample_patient_params(ptype, rng)
                sol    = simulate_patient(params)

                # skip failed simulations
                if string(sol.retcode) ∉ ("Success", "Default")
                    continue
                end

                feats        = extract_features(sol; rng=rng)
                g_sparse, i_sparse = extract_sparse_measurements(sol; rng=rng)
                if any(isnan, feats) || any(isinf, feats) ||
                   any(isnan, g_sparse) || any(isnan, i_sparse)
                    continue
                end

                push!(all_features, feats)
                push!(all_glucose,  g_sparse)
                push!(all_insulin,  i_sparse)
                push!(all_labels, label_map[ptype])
                generated += 1
            catch e
                @debug "Simulation failed for $ptype" exception = e
                continue
            end
        end
        @info "$ptype: generated $generated / $n_per_type"
    end

    features = Matrix(reduce(hcat, all_features)')
    glucose  = Matrix(reduce(hcat, all_glucose)')
    insulin  = Matrix(reduce(hcat, all_insulin)')
    return features, glucose, insulin, all_labels
end

# ─── CSV I/O ─────────────────────────────────────────────────────────────────

"""Write features, sparse measurements, and labels to CSV files in `output_dir`."""
function save_data(output_dir::String, features::Matrix,
                   glucose::Matrix, insulin::Matrix, labels::Vector{Int})
    mkpath(output_dir)

    open(joinpath(output_dir, "features.csv"), "w") do io
        println(io, join(FEATURE_NAMES, ","))
        for i in axes(features, 1)
            println(io, join(features[i, :], ","))
        end
    end

    open(joinpath(output_dir, "sparse_glucose.csv"), "w") do io
        println(io, join(["g_$(t)" for t in GLUCOSE_TIMES], ","))
        for i in axes(glucose, 1)
            println(io, join(glucose[i, :], ","))
        end
    end

    open(joinpath(output_dir, "sparse_insulin.csv"), "w") do io
        println(io, join(["i_$(t)" for t in INSULIN_TIMES], ","))
        for i in axes(insulin, 1)
            println(io, join(insulin[i, :], ","))
        end
    end

    open(joinpath(output_dir, "labels.csv"), "w") do io
        println(io, "label")
        for l in labels
            println(io, l)
        end
    end
    @info "Saved $(size(features, 1)) patients to $output_dir"
end

"""Load features and labels back from CSV (used by gating_network.jl)."""
function load_data(data_dir::String)
    # features
    lines   = readlines(joinpath(data_dir, "features.csv"))
    header  = split(lines[1], ",")
    n       = length(lines) - 1
    nf      = length(header)
    features = zeros(Float64, n, nf)
    for i in 1:n
        features[i, :] .= parse.(Float64, split(lines[i + 1], ","))
    end

    # labels
    llines = readlines(joinpath(data_dir, "labels.csv"))
    labels = [parse(Int, strip(l)) for l in llines[2:end]]

    return features, labels
end

# ─── Main entry point ────────────────────────────────────────────────────────

function main(; n_per_type=100, seed=42)
    output_dir = joinpath(@__DIR__, "data")
    @info "Generating mock EDES patient data" n_per_type seed
    features, glucose, insulin, labels = generate_cohort(n_per_type; seed=seed)
    save_data(output_dir, features, glucose, insulin, labels)
    return features, glucose, insulin, labels
end

if abspath(PROGRAM_FILE) == @__FILE__
    npt = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
    main(; n_per_type=npt)
end
