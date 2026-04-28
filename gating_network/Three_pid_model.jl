using StaticArrays
using OrdinaryDiffEq
using Flux

# ─── 1. Load the Gating Network ──────────────────────────────────────────────
include("gating_network.jl")

# ─── 2. The MoE ODE Model ────────────────────────────────────────────────────
function edesode_moe(u, p, t)
    Ggut, Gpl, Ipl, Irem = u
    
    # Unpack base parameters
    k1, k2, k3, k4, k5, k9, k10,
    tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1,
    sigma, Dmeal, bw, Gb, Ib,
    
    # Unpack MoE weights (from gating network)
    w1, w2, w3, 
    
    # Unpack Expert 1 (Healthy)
    k6_1, k7_1, k8_1, 
    # Unpack Expert 2 (IGT)
    k6_2, k7_2, k8_2, 
    # Unpack Expert 3 (T2D)
    k6_3, k7_3, k8_3 = p

    # gut glucose (Weibull meal appearance)
    dGgut = sigma * k1^sigma * t^(sigma - 1) * exp(-(k1 * t)^sigma) * Dmeal - k2 * Ggut

    # plasma glucose
    gliv  = EGPb - k3 * (Gpl - Gb) - k4 * beta * Irem
    ggut  = k2 * (f / (Vg * bw)) * Ggut
    u_ii  = EGPb * ((Km + Gb) / Gb) * (Gpl / (Km + Gpl))
    u_id  = k5 * beta * Irem * (Gpl / (Km + Gpl))
    u_ren = c1 / (Vg * bw) * (Gpl - Gren) * (Gpl > Gren)
    dGpl  = gliv + ggut - u_ii - u_id - u_ren

    # ─── Mixture of Experts Blending ─────────────────────────────────
    
    # Expert 1 (Healthy)
    i_pnc_1 = beta^(-1) * (k6_1 * (Gpl - Gb) + (k7_1 / tau_i) * Gb + k8_1 * tau_d * dGpl)
    i_liv_1 = k7_1 * Gb * Ipl / (beta * tau_i * Ib)
    
    # Expert 2 (IGT)
    i_pnc_2 = beta^(-1) * (k6_2 * (Gpl - Gb) + (k7_2 / tau_i) * Gb + k8_2 * tau_d * dGpl)
    i_liv_2 = k7_2 * Gb * Ipl / (beta * tau_i * Ib)
    
    # Expert 3 (T2D)
    i_pnc_3 = beta^(-1) * (k6_3 * (Gpl - Gb) + (k7_3 / tau_i) * Gb + k8_3 * tau_d * dGpl)
    i_liv_3 = k7_3 * Gb * Ipl / (beta * tau_i * Ib)

    # Blend the control actions based on the gating network weights
    i_pnc = (w1 * i_pnc_1) + (w2 * i_pnc_2) + (w3 * i_pnc_3)
    i_liv = (w1 * i_liv_1) + (w2 * i_liv_2) + (w3 * i_liv_3)
    # ─────────────────────────────────────────────────────────────────

    i_int = k9 * (Ipl - Ib)
    dIpl  = i_pnc - i_liv - i_int
    dIrem = i_int - k10 * Irem

    return SA[dGgut, dGpl, dIpl, dIrem]
end

# ─── 3. Parameter Builder for MoE ────────────────────────────────────────────
function build_moe_parameters(k1, k5, Gb, Ib, BW, w, expert_gains; Dmeal=75_000.0)
    base_params = (
        k1, 0.28, 6.07e-3, 2.35e-4, k5, 3.83e-2, 2.84e-1, 
        31.0, 3.0, 1.0, 9.0, 0.043, 13.2, 0.005551, 17.0/70.0, 0.1, 
        1.4, Dmeal, BW, Gb, Ib 
    )
    
    return (
        base_params..., 
        w[1], w[2], w[3],
        expert_gains.healthy..., 
        expert_gains.igt..., 
        expert_gains.t2d...
    )
end

# ─── 4. Full Simulation Pipeline ─────────────────────────────────────────────
function simulate_patient_moe(patient_features, patient_base_params, predictor, expert_gains, tgrid)
    w = predict_gates(predictor, patient_features)
    @info "Gating Weights" Healthy=round(w[1], digits=3) IGT=round(w[2], digits=3) T2D=round(w[3], digits=3)
    
    k1, k5, Gb, Ib, BW = patient_base_params
    p_moe = build_moe_parameters(k1, k5, Gb, Ib, BW, w, expert_gains)
    
    u0    = SA[0.0, Gb, Ib, 0.0]
    tspan = (Float64(first(tgrid)), Float64(last(tgrid)))
    prob  = ODEProblem(edesode_moe, u0, tspan, p_moe)
    
    sol = solve(prob, Tsit5(); saveat=collect(tgrid), reltol=1e-5, abstol=1e-8)
    
    G_sim = [sol.u[i][2] for i in eachindex(sol.t)]
    I_sim = [sol.u[i][3] for i in eachindex(sol.t)]
    
    return G_sim, I_sim
end

# ─── 5. Execution Block (Runs automatically on include) ──────────────────────

# Point to the data directory (since we are inside the gating_network folder)
data_dir = joinpath(@__DIR__, "data")
predictor, _ = train_and_build(data_dir)

# Pre-tuned expert parameters (k6, k7, k8)
tuned_experts = (
    healthy = (0.005, 1.15, 0.5),   
    igt     = (0.003, 1.10, 0.8),   
    t2d     = (0.001, 1.05, 1.2)    
)

# Mock patient data
mock_features = [
    6.0,      # 1. fasting_glucose (Matches Gb)
    10.5,     # 2. peak_glucose (Higher peak)
    60.0,     # 3. time_peak_glucose (Slightly delayed)
    80.5,      # 4. glucose_120 (CRITICAL: between 7.8 and 11.1 for IGT)
    2000.0,   # 5. glucose_auc 
    12.0,     # 6. fasting_insulin (Matches Ib, slightly elevated)
    80.0,     # 7. peak_insulin (Compensatory hyperinsulinemia)
    90.0,     # 8. time_peak_insulin (Delayed insulin response)
    40.0,     # 9. insulin_30
    60.0,     # 10. insulin_120
    12000.0,  # 11. insulin_auc
    3.2       # 12. homa_ir (Calculated as 6.0 * 12.0 / 22.5)
]
mock_base_params = (0.02, 0.05, 5.0, 5.0, 75.0) 
tgrid = 0.0:1.0:240.0

# Run simulation
G_out, I_out = simulate_patient_moe(mock_features, mock_base_params, predictor, tuned_experts, tgrid)

println("\nSimulation complete!")
println("  Peak Plasma Glucose: $(round(maximum(G_out), digits=2)) mmol/L")
println("  Peak Plasma Insulin: $(round(maximum(I_out), digits=2)) mU/L")