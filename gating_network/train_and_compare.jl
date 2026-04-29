#=
    Train both gating-network variants on sparse mock data, evaluate on the
    50-patient real juliacon-2024 cohort, optionally fine-tune on real data,
    and produce 3D + ternary weight-scatter plots coloured by Matsuda index.

    Usage (from the gating_network/ directory):
      julia --project mock_data_generation.jl 200      # generate sparse mock data
      julia --project train_and_compare.jl              # train + evaluate + plot
=#

using Statistics
using Printf
using Flux
using Flux: onehotbatch, onecold
using Random

include("gating_network.jl")
include("real_data.jl")
include("plot_weights.jl")

const DATA_DIR    = joinpath(@__DIR__, "data")
const PREDICT_DIR = joinpath(@__DIR__, "..", "juliacon-2024",
                             "2_parameter_estimation", "predict")
const FIG_DIR     = joinpath(@__DIR__, "figures")

# ─── Evaluation helpers ─────────────────────────────────────────────────────

"""Class-wise accuracy on a held-out cohort."""
function evaluate_on(gp::GatingPredictor, X::AbstractMatrix, y::AbstractVector{Int})
    W = predict_gates_batch(gp, X)
    ŷ = vec(map(argmax, eachcol(W)))
    overall = mean(ŷ .== y)
    per_class = Dict{Int,Float64}()
    for c in 1:3
        idx = findall(==(c), y)
        per_class[c] = isempty(idx) ? NaN : mean(ŷ[idx] .== c)
    end
    return (overall=overall, per_class=per_class, ŷ=ŷ, W=W)
end

function _confusion_block(ŷ, y; title)
    println(title)
    println("                ", join(lpad.(["Healthy", "IGT", "T2D"], 9)))
    names = ["Healthy", "    IGT", "    T2D"]
    for i in 1:3
        row = [count((ŷ .== j) .& (y .== i)) for j in 1:3]
        println("    $(names[i])  ", join(lpad.(row, 9)))
    end
    println()
end

"""
    fine_tune(gp, X, y; epochs, lr) -> GatingPredictor

Continue training an existing predictor on a small held-out cohort.
Re-uses the predictor's z-norm stats (don't recompute on tiny n).
"""
function fine_tune(gp::GatingPredictor, X::AbstractMatrix, y::AbstractVector{Int};
        epochs::Int=300, lr::Float64=1e-4, seed::Int=0)
    Random.seed!(seed)
    Xn   = (Float32.(X') .- gp.μ) ./ gp.σ
    yoh  = onehotbatch(y, 1:3)
    cw   = compute_class_weights(y, 3)

    model = deepcopy(gp.model)
    opt   = Flux.setup(Adam(lr), model)
    Flux.trainmode!(model)
    for _ in 1:epochs
        _, grads = Flux.withgradient(model) do m
            weighted_logitcrossentropy(m(Xn), yoh, cw)
        end
        Flux.update!(opt, model, grads[1])
    end
    Flux.testmode!(model)
    return GatingPredictor(model, gp.μ, gp.σ, gp.mode)
end

# ─── Run ─────────────────────────────────────────────────────────────────────

function run(; do_fine_tune::Bool=true, save_figs::Bool=true)
    isfile(joinpath(DATA_DIR, "sparse_glucose.csv")) ||
        error("No sparse mock data found in $DATA_DIR. " *
              "Run `julia mock_data_generation.jl 200` first.")
    isdir(PREDICT_DIR) ||
        error("Real data dir $PREDICT_DIR not found")

    @info "Loading real cohort"
    real = real_dataset(PREDICT_DIR)
    println("\nReal cohort label distribution: ",
            "Healthy=$(count(==(1), real.labels)) ",
            "IGT=$(count(==(2), real.labels)) ",
            "T2D=$(count(==(3), real.labels))")
    println("Matsuda quartiles: ",
            round.(quantile(real.matsuda, [0.0, 0.25, 0.5, 0.75, 1.0]); digits=2))

    save_figs && mkpath(FIG_DIR)

    results = Dict{Symbol,Any}()

    for mode in (:glucose, :glucose_insulin)
        println("\n══════════════════════════════════════════════")
        println("  Training variant: $mode")
        println("══════════════════════════════════════════════")
        gp, history = train_and_build(DATA_DIR; mode=mode)

        # held-out real-cohort evaluation
        X_real = mode === :glucose ? real.X_glucose : real.X_glu_ins
        ev = evaluate_on(gp, X_real, real.labels)
        @printf("\n[%s] real-cohort accuracy: %.1f%%   (Healthy %.1f%% / IGT %.1f%% / T2D %.1f%%)\n",
                mode, 100ev.overall,
                100ev.per_class[1], 100ev.per_class[2], 100ev.per_class[3])
        _confusion_block(ev.ŷ, real.labels;
                         title="\n[$mode] real-cohort confusion (rows=actual, cols=pred)")

        gp_used = gp
        if do_fine_tune
            gp_ft = fine_tune(gp, X_real, real.labels)
            ev_ft = evaluate_on(gp_ft, X_real, real.labels)
            @printf("[%s] AFTER fine-tune: %.1f%%   (Healthy %.1f%% / IGT %.1f%% / T2D %.1f%%)\n",
                    mode, 100ev_ft.overall,
                    100ev_ft.per_class[1], 100ev_ft.per_class[2], 100ev_ft.per_class[3])
            _confusion_block(ev_ft.ŷ, real.labels;
                             title="[$mode] after fine-tune confusion")
            gp_used = gp_ft   # use the fine-tuned predictor for plotting
        end

        # 3D + ternary plot of expert weights coloured by Matsuda
        W = predict_gates_batch(gp_used, X_real)
        figpath = save_figs ? joinpath(FIG_DIR, "weights_$(mode).png") : nothing
        fig = plot_weights_3d(W, real.matsuda;
            title="gating weights — $mode" * (do_fine_tune ? " (fine-tuned)" : ""),
            savepath=figpath)

        results[mode] = (gp=gp_used, history=history, eval=ev, figure=fig)
    end

    println("\nDone. Figures saved under $FIG_DIR" * (save_figs ? "" : " (skipped)"))
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
