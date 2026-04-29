#=
    Real sparse OGTT data loader
    ============================
    Reads the 50-patient cohort in juliacon-2024/2_parameter_estimation/predict/,
    derives ADA labels from fasting + 2-h glucose, and computes the Matsuda
    insulin-sensitivity index from the same sparse measurements.

    Real OGTT schedule (matches GLUCOSE_TIMES / INSULIN_TIMES in mock_data_generation.jl):
      glucose: 0, 15, 30, 60, 120, 180, 240 min  (7 points)
      insulin: 0, 15, 30, 60, 120,      240 min  (6 points — no 180)

    Files are semicolon-separated, no header row.
=#

using Statistics

const REAL_GLUCOSE_TIMES = (0, 15, 30, 60, 120, 180, 240)
const REAL_INSULIN_TIMES = (0, 15, 30, 60, 120, 240)

"""
    load_real_sparse(predict_dir) -> (glucose::Matrix, insulin::Matrix)

Returns (n_patients × 7) glucose and (n_patients × 6) insulin matrices.
"""
function load_real_sparse(predict_dir::String)
    glucose = _read_semicolon_matrix(joinpath(predict_dir, "glucose.csv"))
    insulin = _read_semicolon_matrix(joinpath(predict_dir, "insulin.csv"))
    size(glucose, 1) == size(insulin, 1) ||
        error("glucose ($(size(glucose,1)) rows) and insulin ($(size(insulin,1)) rows) disagree")
    size(glucose, 2) == length(REAL_GLUCOSE_TIMES) ||
        error("expected $(length(REAL_GLUCOSE_TIMES)) glucose columns, got $(size(glucose,2))")
    size(insulin, 2) == length(REAL_INSULIN_TIMES) ||
        error("expected $(length(REAL_INSULIN_TIMES)) insulin columns, got $(size(insulin,2))")
    return glucose, insulin
end

function _read_semicolon_matrix(path::String)
    lines = filter(!isempty, strip.(readlines(path)))
    rows  = [parse.(Float64, split(l, ';')) for l in lines]
    n     = length(rows)
    m     = length(rows[1])
    M     = zeros(Float64, n, m)
    for i in 1:n
        length(rows[i]) == m ||
            error("row $i in $path has $(length(rows[i])) columns, expected $m")
        M[i, :] .= rows[i]
    end
    return M
end

# ─── ADA labelling ──────────────────────────────────────────────────────────
# 3-class scheme used throughout this project:
#   Healthy: FPG < 5.6  AND  2-h glucose < 7.8       (mmol/L)
#   T2D    : FPG ≥ 7.0  OR   2-h glucose ≥ 11.1
#   IGT    : everything in between (covers IFG and "classical" IGT)

"""
    ada_label(fpg, g_2h) -> Int

Apply ADA criteria to a single patient's fasting and 2-h glucose values.
Returns 1 = Healthy, 2 = IGT, 3 = T2D.
"""
function ada_label(fpg::Real, g_2h::Real)
    if fpg ≥ 7.0 || g_2h ≥ 11.1
        return 3
    elseif fpg < 5.6 && g_2h < 7.8
        return 1
    else
        return 2
    end
end

"""
    ada_labels(glucose) -> Vector{Int}

Apply ADA criteria to a (n_patients × 7) glucose matrix sampled at REAL_GLUCOSE_TIMES.
"""
function ada_labels(glucose::AbstractMatrix)
    fpg_idx  = findfirst(==(0),   REAL_GLUCOSE_TIMES)
    g120_idx = findfirst(==(120), REAL_GLUCOSE_TIMES)
    return [ada_label(glucose[i, fpg_idx], glucose[i, g120_idx])
            for i in axes(glucose, 1)]
end

# ─── Matsuda index ──────────────────────────────────────────────────────────
# Matsuda et al. (1999): combines fasting and post-load glucose & insulin.
#
#   ISI_Matsuda = 10000 / sqrt( G0 · I0 · Ḡ · Ī )
#
# where Ḡ and Ī are the means of the OGTT samples. Higher = more insulin-
# sensitive. Healthy values are typically > ~4; values < ~2.5 suggest IR.
# We require glucose in mg/dL and insulin in µU/mL — the juliacon-2024 data
# is stored as mmol/L glucose and mU/L insulin, so we convert glucose using
# 1 mmol/L = 18.0182 mg/dL (insulin µU/mL ≡ mU/L numerically).

const MMOL_TO_MGDL = 18.0182

"""
    matsuda_index(glucose_mmol, insulin_mUL) -> Float64

`glucose_mmol` and `insulin_mUL` are vectors of co-located OGTT samples
(must share the same time grid; pad insulin to the glucose grid first if
necessary). Returns the unitless Matsuda index.
"""
function matsuda_index(glucose_mmol::AbstractVector, insulin_mUL::AbstractVector)
    length(glucose_mmol) == length(insulin_mUL) ||
        error("matsuda_index: glucose and insulin vectors must share length")
    g_mgdl = glucose_mmol .* MMOL_TO_MGDL
    G0, I0 = g_mgdl[1], insulin_mUL[1]
    Gbar   = mean(g_mgdl)
    Ibar   = mean(insulin_mUL)
    return 10000.0 / sqrt(max(G0 * I0 * Gbar * Ibar, eps()))
end

"""
    matsuda_indices(glucose, insulin; glucose_times, insulin_times) -> Vector{Float64}

Compute Matsuda for a whole cohort. Insulin is interpolated onto the glucose
time grid (linear, edge-clamped) so the two share a common schedule before
averaging. Defaults match the OGTT schedules used in this project.
"""
function matsuda_indices(glucose::AbstractMatrix, insulin::AbstractMatrix;
        glucose_times=REAL_GLUCOSE_TIMES,
        insulin_times=REAL_INSULIN_TIMES)
    n = size(glucose, 1)
    n == size(insulin, 1) || error("row count mismatch")
    out = zeros(Float64, n)
    for i in 1:n
        i_on_g = interp_to_grid(collect(insulin_times), insulin[i, :], collect(glucose_times))
        out[i] = matsuda_index(glucose[i, :], i_on_g)
    end
    return out
end

"""Linear interpolation of `y(x)` onto `xq`, clamping outside the input range."""
function interp_to_grid(x::AbstractVector, y::AbstractVector, xq::AbstractVector)
    length(x) == length(y) || error("x and y must have equal length")
    out = similar(xq, Float64)
    for (k, q) in enumerate(xq)
        if q ≤ x[1]
            out[k] = y[1]
        elseif q ≥ x[end]
            out[k] = y[end]
        else
            j = searchsortedlast(x, q)
            t = (q - x[j]) / (x[j+1] - x[j])
            out[k] = (1 - t) * y[j] + t * y[j+1]
        end
    end
    return out
end

# ─── Convenience bundle ─────────────────────────────────────────────────────

"""
    real_dataset(predict_dir) -> NamedTuple

One-call loader returning everything downstream code needs:
  glucose, insulin, labels, matsuda, X_glucose (= glucose), X_glu_ins (= [glucose insulin]).
"""
function real_dataset(predict_dir::String)
    glucose, insulin = load_real_sparse(predict_dir)
    labels   = ada_labels(glucose)
    matsuda  = matsuda_indices(glucose, insulin)
    return (
        glucose   = glucose,
        insulin   = insulin,
        labels    = labels,
        matsuda   = matsuda,
        X_glucose = glucose,
        X_glu_ins = hcat(glucose, insulin),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    predict_dir = joinpath(@__DIR__, "..", "juliacon-2024", "2_parameter_estimation", "predict")
    ds = real_dataset(predict_dir)
    println("Loaded $(size(ds.glucose, 1)) real patients")
    println("Label distribution: ",
            Dict("Healthy" => count(==(1), ds.labels),
                 "IGT"     => count(==(2), ds.labels),
                 "T2D"     => count(==(3), ds.labels)))
    println("Matsuda summary: min=$(round(minimum(ds.matsuda); digits=2)) " *
            "median=$(round(median(ds.matsuda); digits=2)) " *
            "max=$(round(maximum(ds.matsuda); digits=2))")
end
