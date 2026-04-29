#=
    3D scatter of gating-network expert weights, coloured by Matsuda index
    ======================================================================
    Each patient's softmax output (w_healthy, w_IGT, w_T2D) is a point on
    the 2-simplex inside the unit cube. We plot:
      (left)  the 3D scatter inside the simplex triangle, and
      (right) a 2D ternary projection of the same points
    so the cluster structure is readable both ways.

    Colour encodes the Matsuda insulin-sensitivity index — high (sensitive)
    points should cluster near the "Healthy" vertex, low points near "T2D".

    Usage:
      include("gating_network.jl")
      include("real_data.jl")
      include("plot_weights.jl")

      ds  = real_dataset(joinpath(@__DIR__, "..", "juliacon-2024",
                                  "2_parameter_estimation", "predict"))
      gp, _ = train_and_build(joinpath(@__DIR__, "data"); mode=:glucose_insulin)
      W = predict_gates_batch(gp, ds.X_glu_ins)              # 3 × n
      plot_weights_3d(W, ds.matsuda; title="glucose+insulin")
=#

using GLMakie
using Statistics

"""
    plot_weights_3d(W, matsuda; title, savepath=nothing) -> Figure

`W` is a (3 × n_patients) matrix of softmax weights (rows = healthy/IGT/T2D),
`matsuda` is a length-n vector of Matsuda indices.
"""
function plot_weights_3d(W::AbstractMatrix, matsuda::AbstractVector;
        title::String="gating-network weights",
        savepath::Union{Nothing,String}=nothing,
        cmap=:viridis)

    size(W, 1) == 3 ||
        error("expected W to be (3 × n_patients), got $(size(W))")
    size(W, 2) == length(matsuda) ||
        error("W has $(size(W,2)) patients but matsuda has $(length(matsuda))")

    w1, w2, w3 = W[1, :], W[2, :], W[3, :]

    # log-scale Matsuda — its distribution is heavily right-skewed in real cohorts
    c = log10.(max.(matsuda, 1e-3))

    fig = Figure(size = (1280, 560))
    Label(fig[0, :], title; fontsize = 18, halign = :center)

    # ─── 3D simplex ──────────────────────────────────────────────────────
    ax3 = Axis3(fig[1, 1];
        xlabel = "w_Healthy", ylabel = "w_IGT", zlabel = "w_T2D",
        title  = "weights on 2-simplex",
        aspect = (1, 1, 1),
        limits = ((0, 1), (0, 1), (0, 1)),
        viewmode = :fit)

    # draw the simplex triangle (vertices at the three unit axes) for context
    tri = [Point3f(1,0,0), Point3f(0,1,0), Point3f(0,0,1), Point3f(1,0,0)]
    lines!(ax3, tri; color = (:gray, 0.6), linewidth = 1.5)
    text!(ax3, Point3f(1.02, 0, 0); text = "Healthy", align = (:left, :center))
    text!(ax3, Point3f(0, 1.02, 0); text = "IGT",     align = (:left, :center))
    text!(ax3, Point3f(0, 0, 1.02); text = "T2D",     align = (:left, :center))

    sc = scatter!(ax3, w1, w2, w3;
        color = c, colormap = cmap, markersize = 10,
        strokecolor = :black, strokewidth = 0.3)

    # ─── 2D ternary projection ───────────────────────────────────────────
    # Standard ternary embedding: x = w_IGT + w_T2D / 2, y = w_T2D · √3/2.
    ax2 = Axis(fig[1, 2];
        xlabel = "", ylabel = "",
        title  = "ternary projection",
        aspect = AxisAspect(1))
    hidedecorations!(ax2)
    hidespines!(ax2)

    tx = w2 .+ w3 ./ 2
    ty = w3 .* (sqrt(3) / 2)

    # triangle frame
    tri2 = [Point2f(0, 0), Point2f(1, 0), Point2f(0.5, sqrt(3)/2), Point2f(0, 0)]
    lines!(ax2, tri2; color = (:gray, 0.6), linewidth = 1.5)
    text!(ax2, Point2f(-0.02, -0.03); text = "Healthy", align = (:right, :top))
    text!(ax2, Point2f( 1.02, -0.03); text = "IGT",     align = (:left,  :top))
    text!(ax2, Point2f( 0.50,  sqrt(3)/2 + 0.02); text = "T2D",
          align = (:center, :bottom))

    scatter!(ax2, tx, ty;
        color = c, colormap = cmap, markersize = 10,
        strokecolor = :black, strokewidth = 0.3)

    Colorbar(fig[1, 3], sc;
        label = "log10(Matsuda index)", height = Relative(0.7))

    if savepath !== nothing
        save(savepath, fig)
        @info "saved figure to $savepath"
    end
    return fig
end

"""
    plot_weights_for_predictor(gp, X, matsuda; kw...) -> Figure

Convenience wrapper: runs `predict_gates_batch` on `X` and plots.
"""
function plot_weights_for_predictor(gp, X::AbstractMatrix,
                                    matsuda::AbstractVector; kw...)
    W = predict_gates_batch(gp, X)
    return plot_weights_3d(W, matsuda; kw...)
end
