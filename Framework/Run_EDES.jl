# Run_EDES.jl
using DelayDiffEq
using Plots

include("EDES_Types.jl")
include("EDES_Core.jl")

# 1. Initialize structs
constants = EDESConstants()
inputs    = EDESInputs(D_meal_G=75_000.0, BW=75.0)
params    = EDESParameters()

# 2. Instantiate chosen controller (PID for exact MATLAB reproduction)

pid_ctrl = PIDController(0.092, 0.079, 7.394) #Healthy
#pid_ctrl = PIDController(0.006, 0.089, 4.724) #IGT
#pid_ctrl = PIDController(0.014, 0.000, 5.755) #T2D

# Group them into a single tuple for the solver
p_system = (params, constants, inputs, pid_ctrl)

# 3. Setup initial conditions and time span
u0    = [0.0, params.G_b, 0.0, params.I_PL_b, 0.0]
tspan = (0.0, 240.0)

# 4. Define and solve the DDE Problem
prob = DDEProblem(edes_dde_modular!, u0, history_func, tspan, p_system)

# MethodOfSteps(Tsit5()) is highly efficient for biological DDEs
sol = solve(prob, MethodOfSteps(Tsit5()), reltol=1e-5, abstol=1e-8)

# 5. Extract results exactly on the minute
t_steps = 0.0:1.0:240.0
G_sim   = [sol(t)[2] for t in t_steps]
I_sim   = [sol(t)[4] for t in t_steps]

# ─── 6. Visualization ─────────────────────────────────────────────────────────

p_glucose = plot(t_steps, G_sim, 
    label="Plasma Glucose", 
    color=:blue, 
    linewidth=2.5, 
    ylabel="Glucose (mmol/L)", 
    title="Postprandial Response"
)

p_insulin = plot(t_steps, I_sim, 
    label="Plasma Insulin", 
    color=:red, 
    linewidth=2.5, 
    xlabel="Time (min)", 
    ylabel="Insulin (mU/L)"
)

final_figure = plot(p_glucose, p_insulin, layout=(2, 1), size=(800, 600), margin=5Plots.mm)
display(final_figure)