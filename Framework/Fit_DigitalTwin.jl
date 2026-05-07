# Fit_MoE_Twin.jl
using OrdinaryDiffEq
using Optimization
using OptimizationOptimJL
using Plots
using Statistics

include("EDES_Types.jl")
include("EDES_Core.jl")

# ─── 1. Sparse Clinical Data (11 points out to 480 mins) ─────────────────────
t_data = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 360.0, 480.0]
G_data = [5.0, 8.5, 7.8, 6.8, 6.2, 5.8, 5.5, 5.2, 5.1, 5.0, 5.0]  
I_data = [5.0, 55.0, 45.0, 35.0, 25.0, 18.0, 12.0, 8.0, 6.0, 5.0, 5.0] 

# ─── 2. Fixed System Setup & Pre-tuned Experts ───────────────────────────────
constants = EDESConstants()
inputs    = EDESInputs(D_meal_G=75_000.0, BW=75.0)
tspan     = (0.0, 480.0) 

# Define your 3 clinical phenotypes (PID parameters: k6, k7, k8)
healthy_expert = PIDController(0.092, 0.079, 7.394)     # Healthy
igt_expert     = PIDController(0.006, 0.089, 4.724)        # IGT
t2d_expert     = PIDController(0.014, 0.000, 5.755)        # T2D

# ─── 3. The Loss Function ────────────────────────────────────────────────────
function moe_twin_loss(x, p_fixed)
    # The optimizer guesses biology (k1, k5) and gating weights (w1, w2, w3)
    opt_k1, opt_k5, w1, w2, w3 = x
    
    # Force weights to be positive and sum to 1.0 (Softmax-like behavior)
    abs_w1, abs_w2, abs_w3 = abs(w1), abs(w2), abs(w3)
    total_w = abs_w1 + abs_w2 + abs_w3
    norm_weights = [abs_w1/total_w, abs_w2/total_w, abs_w3/total_w]
    
    # Build the parameters and the MoE Controller
    params = EDESParameters(k1=abs(opt_k1), k5=abs(opt_k5))
    moe_ctrl = MoEController(healthy_expert, igt_expert, t2d_expert, norm_weights)
    
    u0 = SA[0.0, G_data[1], 0.0, I_data[1], 0.0, G_data[1], G_data[1], G_data[1]]
    
    prob = ODEProblem(edes_ode_modular, u0, tspan, (params, constants, inputs, moe_ctrl))
    sol = solve(prob, Tsit5(), saveat=t_data, verbose=false)
    
    if sol.retcode != ReturnCode.Success
        return Inf 
    end
    
    G_sim = [u[2] for u in sol.u]
    I_sim = [u[4] for u in sol.u]
    
    G_error = mean((G_sim .- G_data).^2) * 10.0 
    I_error = mean((I_sim .- I_data).^2)
    
    return G_error + I_error
end

# ─── 4. Run the Optimizer ────────────────────────────────────────────────────
# Initial Guesses: [k1, k5, w_healthy, w_igt, w_t2d]
# We start by guessing an equal 33% split between the experts
x0 = [0.02, 0.05, 0.33, 0.33, 0.33] 

println("Fitting MoE Digital Twin...")

optf = OptimizationFunction(moe_twin_loss)
optprob = OptimizationProblem(optf, x0)
res = solve(optprob, NelderMead(), maxiters=3000)

# Process final parameters
opt_x = abs.(res.u)
final_total_w = opt_x[3] + opt_x[4] + opt_x[5]
final_weights = [opt_x[3]/final_total_w, opt_x[4]/final_total_w, opt_x[5]/final_total_w]

println("\n--- MoE Patient Profile ---")
println("Gastric Emptying (k1) : ", round(opt_x[1], digits=4))
println("Insulin Sensitivity (k5): ", round(opt_x[2], digits=4))
println("Healthy Mixture  : ", round(final_weights[1]*100, digits=1), "%")
println("IGT Mixture      : ", round(final_weights[2]*100, digits=1), "%")
println("T2D Mixture      : ", round(final_weights[3]*100, digits=1), "%")

# ─── 5. Visualize ────────────────────────────────────────────────────────────
final_params = EDESParameters(k1=opt_x[1], k5=opt_x[2])
final_ctrl   = MoEController(healthy_expert, igt_expert, t2d_expert, final_weights)
final_u0     = SA[0.0, G_data[1], 0.0, I_data[1], 0.0, G_data[1], G_data[1], G_data[1]]

twin_prob = ODEProblem(edes_ode_modular, final_u0, (0.0, 480.0), (final_params, constants, inputs, final_ctrl))
twin_sol  = solve(twin_prob, Tsit5(), saveat=1.0)

t_steps = twin_sol.t
G_twin  = [u[2] for u in twin_sol.u]
I_twin  = [u[4] for u in twin_sol.u]

p_g = plot(t_steps, G_twin, label="MoE Fit", color=:blue, linewidth=2, title="Glucose Twin (MoE)")
scatter!(p_g, t_data, G_data, label="Real Data", color=:black)

p_i = plot(t_steps, I_twin, label="MoE Fit", color=:red, linewidth=2, title="Insulin Twin (MoE)")
scatter!(p_i, t_data, I_data, label="Real Data", color=:black)

display(plot(p_g, p_i, layout=(2,1), size=(800,600)))