#=
    Gating Network for Mixture of Experts (MoE)
    ============================================
    Classifies patients into 3 metabolic types from a sparse OGTT meal response:
      1 = Healthy  (normal glucose tolerance)
      2 = IGT      (impaired glucose tolerance)
      3 = T2D      (type 2 diabetes)

    Three input modes are supported (selected via `mode` keyword):

      :glucose          → 7 sparse glucose readings   (t = 0,15,30,60,120,180,240)
      :glucose_insulin  → 7 glucose + 6 insulin       (insulin t = 0,15,30,60,120,240)
      :features         → 12 derived features         (legacy path; for comparison)

    Sparse modes match the real OGTT sampling schedule used in
    juliacon-2024/2_parameter_estimation/predict/, so the gating network is
    trained on the same kind of measurements it will see at inference time.

    Usage:
      include("gating_network.jl")
      # generate sparse training data first:
      #   julia mock_data_generation.jl 200
      gp_g,  history_g  = train_and_build(joinpath(@__DIR__, "data"); mode=:glucose)
      gp_gi, history_gi = train_and_build(joinpath(@__DIR__, "data"); mode=:glucose_insulin)
=#

using Flux
using Flux: onehotbatch, logitcrossentropy, onecold
using Statistics
using Random
using Printf

# ─── Data helpers ─────────────────────────────────────────────────────────────

"""Read a comma-separated CSV with a single header row into a Matrix{Float64}."""
function _read_matrix(path::String)
    lines  = readlines(path)
    header = split(lines[1], ",")
    n, nf  = length(lines) - 1, length(header)
    M      = zeros(Float64, n, nf)
    for i in 1:n
        M[i, :] .= parse.(Float64, split(lines[i + 1], ","))
    end
    return M
end

"""
    load_data(data_dir; mode)

Load the input matrix selected by `mode` plus the integer label vector.
"""
function load_data(data_dir::String; mode::Symbol=:glucose)
    if mode === :features
        X = _read_matrix(joinpath(data_dir, "features.csv"))
    elseif mode === :glucose
        X = _read_matrix(joinpath(data_dir, "sparse_glucose.csv"))
    elseif mode === :glucose_insulin
        G = _read_matrix(joinpath(data_dir, "sparse_glucose.csv"))
        I = _read_matrix(joinpath(data_dir, "sparse_insulin.csv"))
        size(G, 1) == size(I, 1) ||
            error("sparse_glucose.csv and sparse_insulin.csv have different row counts")
        X = hcat(G, I)
    else
        error("Unknown mode $mode (expected :glucose, :glucose_insulin, or :features)")
    end

    llines = readlines(joinpath(data_dir, "labels.csv"))
    labels = [parse(Int, strip(l)) for l in llines[2:end]]
    return X, labels
end

"""Z-score normalisation. Returns normalised data + stats for reuse."""
function znorm(X::AbstractMatrix; μ=nothing, σ=nothing)
    if isnothing(μ)
        μ = vec(mean(X; dims=1))
        σ = vec(std(X; dims=1))
        σ[σ .== 0] .= 1.0
    end
    X_out = (X .- μ') ./ σ'
    return X_out, μ, σ
end

"""Stratified train/test split keeping class balance."""
function split_data(features, labels; test_frac=0.2, seed=42)
    rng       = MersenneTwister(seed)
    classes   = sort(unique(labels))
    train_idx = Int[]
    test_idx  = Int[]
    for c in classes
        idx     = findall(labels .== c)
        shuffle!(rng, idx)
        n_test  = max(1, round(Int, length(idx) * test_frac))
        append!(test_idx,  idx[1:n_test])
        append!(train_idx, idx[n_test+1:end])
    end
    return (features[train_idx, :], labels[train_idx],
            features[test_idx, :],  labels[test_idx])
end

# ─── Network Architecture ────────────────────────────────────────────────────

"""
    build_gating_network(n_features, n_classes; hidden)

Feedforward classifier. Hidden sizes default to a width that scales with the
input dimensionality so a 7-input glucose model isn't built like a 12-feature
one. Outputs *logits* (softmax applied externally).
"""
function build_gating_network(n_features::Int, n_classes::Int;
        hidden::Vector{Int}=default_hidden(n_features))
    layers = Any[]
    prev = n_features
    for h in hidden
        push!(layers, Dense(prev => h, relu))
        prev = h
    end
    push!(layers, Dense(prev => n_classes))
    return Chain(layers...)
end

default_hidden(n_features::Int) =
    n_features <= 8  ? [32, 16] :
    n_features <= 16 ? [64, 32, 16] :
                       [128, 64, 32]

# ─── Training ────────────────────────────────────────────────────────────────

function accuracy(model, X, y)
    ŷ = onecold(model(X), 1:3)
    return mean(ŷ .== y)
end

"""
    compute_class_weights(labels, n_classes) -> Vector{Float32}

Inverse-frequency weights so minority classes get proportionally higher loss.
Normalised so weights sum to n_classes.
"""
function compute_class_weights(labels, n_classes::Int)
    counts = [Float32(count(==(c), labels)) for c in 1:n_classes]
    inv_freq = 1.0f0 ./ max.(counts, 1.0f0)
    weights = inv_freq ./ sum(inv_freq) .* n_classes
    return weights
end

"""Cross-entropy with per-class weights to handle class imbalance."""
function weighted_logitcrossentropy(logits, targets, class_weights)
    per_sample = -sum(targets .* Flux.logsoftmax(logits); dims=1)
    w = sum(class_weights .* targets; dims=1)
    return mean(per_sample .* w)
end

"""
    train_gating(data_dir; mode, epochs, lr, batch_size, seed)

Full training pipeline. Returns `(model, μ, σ, mode, history)`.
"""
function train_gating(data_dir::String;
        mode::Symbol=:glucose,
        epochs=1000, lr=5e-4, batch_size=32, seed=42,
        verbose::Bool=true)

    features, labels = load_data(data_dir; mode=mode)
    X_tr, y_tr, X_te, y_te = split_data(features, labels; seed=seed)

    X_tr_n, μ, σ = znorm(Float32.(X_tr))
    X_te_n, _, _ = znorm(Float32.(X_te); μ=μ, σ=σ)

    X_tr_t  = Matrix(X_tr_n')
    X_te_t  = Matrix(X_te_n')
    y_tr_oh = onehotbatch(y_tr, 1:3)
    y_te_oh = onehotbatch(y_te, 1:3)

    loader = Flux.DataLoader((X_tr_t, y_tr_oh); batchsize=batch_size, shuffle=true)

    cw = compute_class_weights(y_tr, 3)
    verbose && @info "Class weights" healthy=round(cw[1]; digits=2) IGT=round(cw[2]; digits=2) T2D=round(cw[3]; digits=2)

    model     = build_gating_network(size(features, 2), 3)
    opt_state = Flux.setup(Adam(lr), model)

    history = Dict(
        "train_loss" => Float64[], "test_loss"  => Float64[],
        "train_acc"  => Float64[], "test_acc"   => Float64[],
    )

    best_test_loss = Inf
    best_model     = deepcopy(model)

    for epoch in 1:epochs
        Flux.trainmode!(model)
        for (xb, yb) in loader
            _, grads = Flux.withgradient(model) do m
                weighted_logitcrossentropy(m(xb), yb, cw)
            end
            Flux.update!(opt_state, model, grads[1])
        end

        Flux.testmode!(model)
        trl = weighted_logitcrossentropy(model(X_tr_t), y_tr_oh, cw)
        tel = weighted_logitcrossentropy(model(X_te_t), y_te_oh, cw)
        tra = accuracy(model, X_tr_t, y_tr)
        tea = accuracy(model, X_te_t, y_te)

        push!(history["train_loss"], trl)
        push!(history["test_loss"],  tel)
        push!(history["train_acc"],  tra)
        push!(history["test_acc"],   tea)

        if tel < best_test_loss
            best_test_loss = tel
            best_model     = deepcopy(model)
        end

        if verbose && (epoch % 25 == 0 || epoch == 1)
            @printf("[%s] epoch %3d │ train %.4f  test %.4f │ train %5.1f%%  test %5.1f%%\n",
                    mode, epoch, trl, tel, 100tra, 100tea)
        end

        if tra == 1.0 && tea == 1.0
            verbose && @info "Perfect accuracy reached at epoch $epoch"
            break
        end
    end

    model = best_model
    Flux.testmode!(model)

    final_acc = accuracy(model, X_te_t, y_te)
    verbose && @info "[$mode] final test accuracy: $(round(100final_acc; digits=1))%"
    verbose && print_confusion(model, X_te_t, y_te)

    return model, μ, σ, mode, history
end

# ─── Evaluation ──────────────────────────────────────────────────────────────

function print_confusion(model, X, y)
    ŷ      = onecold(model(X), 1:3)
    names  = ["Healthy", "    IGT", "    T2D"]
    println("\nConfusion matrix (rows = actual, cols = predicted)")
    println("             ", join(lpad.(["Healthy", "IGT", "T2D"], 9)))
    for i in 1:3
        row = [count((ŷ .== j) .& (y .== i)) for j in 1:3]
        println("  $(names[i])  ", join(lpad.(row, 9)))
    end
    println()
end

# ─── GatingPredictor — portable struct for MoE integration ───────────────────

struct GatingPredictor{M}
    model::M
    μ::Vector{Float32}
    σ::Vector{Float32}
    mode::Symbol
end

input_dim(gp::GatingPredictor) = length(gp.μ)

"""
    predict_gates(gp, x) -> Vector{Float32}

Return expert weights [P(healthy), P(IGT), P(T2D)] for one patient.
`x` length must match the predictor's input dim (7, 13, or 12 depending on mode).
"""
function predict_gates(gp::GatingPredictor, x::AbstractVector)
    length(x) == input_dim(gp) ||
        error("expected input of length $(input_dim(gp)) for mode $(gp.mode), got $(length(x))")
    xv  = Float32.(reshape(x, :, 1))
    xn  = (xv .- gp.μ) ./ gp.σ
    Flux.testmode!(gp.model)
    return vec(softmax(gp.model(xn)))
end

"""Batch version: `X` is (n_patients × input_dim)."""
function predict_gates_batch(gp::GatingPredictor, X::AbstractMatrix)
    size(X, 2) == input_dim(gp) ||
        error("expected $(input_dim(gp)) columns for mode $(gp.mode), got $(size(X, 2))")
    Xt = Float32.(X')
    Xn = (Xt .- gp.μ) ./ gp.σ
    Flux.testmode!(gp.model)
    return softmax(gp.model(Xn))
end

"""Hard classification (1/2/3) for a single patient."""
function classify(gp::GatingPredictor, x::AbstractVector)
    return argmax(predict_gates(gp, x))
end

# ─── Convenience: train + wrap ────────────────────────────────────────────────

"""
    train_and_build(data_dir; mode=:glucose, kw...) -> (GatingPredictor, history)
"""
function train_and_build(data_dir::String; mode::Symbol=:glucose, kw...)
    model, μ, σ, m, history = train_gating(data_dir; mode=mode, kw...)
    gp = GatingPredictor(model, Float32.(μ), Float32.(σ), m)
    return gp, history
end

# ─── Main ────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = joinpath(@__DIR__, "data")
    if !isfile(joinpath(data_dir, "sparse_glucose.csv"))
        @error "No sparse data found. Run mock_data_generation.jl first."
        exit(1)
    end

    println("\n══════ Glucose-only variant (7 inputs) ══════")
    gp_g, _  = train_and_build(data_dir; mode=:glucose)

    println("\n══════ Glucose + insulin variant (13 inputs) ══════")
    gp_gi, _ = train_and_build(data_dir; mode=:glucose_insulin)

    # quick demo on the first 5 patients in each mode
    label_names = ["Healthy", "IGT", "T2D"]
    for (gp, name) in [(gp_g, "glucose-only"), (gp_gi, "glucose+insulin")]
        X, labels = load_data(data_dir; mode=gp.mode)
        println("\n── Sample predictions ($name) ──")
        for i in 1:min(5, size(X, 1))
            probs = predict_gates(gp, X[i, :])
            pred  = argmax(probs)
            @printf("Patient %3d │ true: %-7s │ pred: %-7s │ gates: [%.2f, %.2f, %.2f]\n",
                    i, label_names[labels[i]], label_names[pred],
                    probs[1], probs[2], probs[3])
        end
    end
end
