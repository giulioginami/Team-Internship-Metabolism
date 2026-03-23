"""
GenerateAndLabel_VirtualPopulation.jl

Direct Julia translation and merge of:
  Generate_VirtualPopulation.m  +  Label_VirtualPopulation.m

Produces the same two outputs (now as .jld2 files):
  virtual_population.jld2          – raw simulation results
  virtual_population_labelled.jld2 – ADA-labelled + category datasets

EDES model reference: Rozendaal et al. (2018)

──────────────────────────────────────────────────────────────────────────
KEY TRANSLATION NOTES
──────────────────────────────────────────────────────────────────────────
The original MATLAB code uses two global arrays (t_saved, G_PL_saved)
and an ODE output function (integratorfunG) to implement the 30-minute
moving-window integral of G_PL inside the ODE right-hand side.

Mathematically this is a Delay Differential Equation (DDE): the derivative
of G_int at time t depends on G_PL(t − 30).  Julia's DDEProblem handles
this natively via its internal continuous interpolant, eliminating the need
for global state entirely.

History function (t ≤ 0): constant at the fasting initial conditions.
This reproduces the MATLAB fallback G_PL_lowerbound = G_PL_saved(1) = G_b
for t ≤ 30 min, when the delay window has not yet been filled.

──────────────────────────────────────────────────────────────────────────
DEPENDENCIES  (add to Project.toml / install with Pkg.add)
  DifferentialEquations   – DDEProblem, MethodOfSteps, Tsit5
  Random                  – MersenneTwister, randperm, rand, randn
  Statistics              – median (not used in generation but available)
  Printf                  – @printf for formatted output
  JLD2                    – save / load .jld2 files

For MATLAB-compatible .mat output replace JLD2 with MAT and use
  matwrite("virtual_population.mat", Dict("virtual_population" => vp))
──────────────────────────────────────────────────────────────────────────
"""

using DifferentialEquations   # DDEProblem, MethodOfSteps, Tsit5, ReturnCode
using Random                   # MersenneTwister, randperm
using Statistics               # (available for downstream use)
using Printf                   # @printf
using JLD2                     # jldsave, load

# ============================================================================
# SECTION 1 — SETTINGS
# (mirrors Generate_VirtualPopulation.m lines 25-27)
# ============================================================================
const N        = 5000             # number of virtual individuals to attempt
const TIME_SIM = 0.0:1.0:480.0   # simulation time vector [min]  (0:1:480)
const MEAL_G   = 75_000.0         # 75 g OGTT in mg
const SEED     = 42               # random seed for reproducibility (rng(42))

# ============================================================================
# SECTION 2 — PARAMETER BOUNDS FOR LATIN HYPERCUBE SAMPLING
# Order of rows: k1, k5, k6, k8, G_b, I_PL_b, BW
# (mirrors Generate_VirtualPopulation.m lines 34-42)
# ============================================================================
const PARAM_BOUNDS = [
    0.005   0.04 ;   # k1     [1/min]  gastric emptying
    0.0     0.07 ;   # k5     [1/min]  delayed insulin-dependent glucose uptake
    0.1     3.0  ;   # k6     [-]      proportional beta-cell gain
    0.5    15.0  ;   # k8     [-]      derivative beta-cell gain
    3.9    12.0  ;   # G_b    [mmol/L] fasting plasma glucose
    2.0    55.6  ;   # I_PL_b [mU/L]   fasting plasma insulin
   60.0   130.0  ;   # BW     [kg]     body weight
]

# ============================================================================
# SECTION 3 — FIXED PARAMETERS (Rozendaal et al. 2018)
# (mirrors Generate_VirtualPopulation.m lines 66-74)
# ============================================================================
const K2       = 0.28        # glucose appearance from gut       [1/min]
const K3       = 6.07e-3     # hepatic suppression by ΔG         [1/min]
const K4       = 2.35e-4     # hepatic suppression by delayed I  [1/min]
const K7       = 1.15        # integral beta-cell gain            [-]
const K9       = 3.83e-2     # insulin plasma → remote            [1/min]
const K10      = 2.84e-1     # insulin remote degradation         [1/min]
const SIGMA    = 1.4         # meal appearance shape factor       [-]
const KM       = 13.2        # Michaelis-Menten coefficient       [mmol/L]
const G_LIV_B  = 0.043       # basal hepatic glucose release

# Model constants (same for all individuals — mirrors EDES_Initial.m)
const F_G              = 0.005551   # glucose conversion factor  [mmol/mg]
const F_I              = 1.0        # insulin conversion factor
const V_G              = 17.0/70.0  # glucose distribution volume [L/kg]
const TAU_I            = 31.0       # integration time constant   [min]
const TAU_D            = 3.0        # derivative time constant    [min]
const G_TH_PL          = 9.0        # renal extraction threshold  [mmol/L]
const T_INTEGRALWINDOW = 30.0       # moving window width         [min]
const C1               = 0.1        # renal filtration rate       [L/min]

# ============================================================================
# SECTION 4 — LATIN HYPERCUBE SAMPLING
# Equivalent to MATLAB's lhsdesign(n, k) (random LHS, no optimisation).
# Each of the k variables is divided into n equal-probability strata;
# one value is drawn uniformly from each stratum and the strata are
# permuted independently for each variable.
# (mirrors Fit_EDES_LatinHyperCube.m and Generate_VirtualPopulation.m)
# ============================================================================
function lhsdesign(n::Int, k::Int; rng=Random.GLOBAL_RNG)::Matrix{Float64}
    result = zeros(n, k)
    for j in 1:k
        perm         = randperm(rng, n)                # random permutation
        result[:, j] = (perm .- 1 .+ rand(rng, n)) ./ n   # scale to [0,1]
    end
    return result
end

# ============================================================================
# SECTION 5 — EDES DELAY DIFFERENTIAL EQUATION
#
# State vector:  u = [M_G_gut, G_PL, G_int, I_PL, I_d1]
#   1  M_G_gut  glucose mass in gut                 [mg]
#   2  G_PL     plasma glucose concentration        [mmol/L]
#   3  G_int    moving-window integral of G_PL      [mmol/L]
#   4  I_PL     plasma insulin concentration        [mU/L]
#   5  I_d1     delayed / remote insulin            [mU/L]
#
# Parameter tuple p = (params, BW, D_meal_G) where params is the
# 15-element vector [k1,..,k10, sigma, KM, G_b, I_PL_b, G_liv_b].
#
# DELAY: G_int requires G_PL(t − 30).
#   • For t > 30 the DDE solver supplies G_PL(t−30) via h(p, t−30)[2].
#   • For t ≤ 30 the history function returns G_b (fasting value).
#   This exactly replicates the MATLAB fallback:
#     G_PL_lowerbound = G_PL_saved(1)  when t <= t_integralwindow
#
# (mirrors EDES_ODE.m)
# ============================================================================
function edes_dde!(du, u, h, p, t)
    params, BW, D_meal_G = p

    # -- Unpack state variables -------------------------------------------
    M_G_gut = u[1]
    G_PL    = u[2]
    G_int   = u[3]
    I_PL    = u[4]
    I_d1    = u[5]

    # -- Unpack 15-element parameter vector --------------------------------
    k1      = params[1]   # gastric emptying
    k2      = params[2]   # gut → plasma
    k3      = params[3]   # hepatic suppression by ΔG
    k4      = params[4]   # hepatic suppression by delayed I
    k5      = params[5]   # insulin-dependent uptake
    k6      = params[6]   # proportional beta-cell gain
    k7      = params[7]   # integral beta-cell gain
    k8      = params[8]   # derivative beta-cell gain
    k9      = params[9]   # plasma → remote insulin
    k10     = params[10]  # remote insulin degradation
    sigma_p = params[11]  # meal appearance shape
    KM_p    = params[12]  # Michaelis-Menten coefficient
    G_b     = params[13]  # basal (fasting) plasma glucose
    I_PL_b  = params[14]  # basal (fasting) plasma insulin
    G_liv_b = params[15]  # basal hepatic glucose release

    # -- Delayed G_PL (the 30-min window for the integral PID term) --------
    # h(p, t - 30) returns the DDE history at (t − 30):
    #   • if t − 30 < 0  → constant history = initial conditions → G_b
    #   • if t − 30 ≥ 0  → DDE solver's continuous interpolant
    if t > T_INTEGRALWINDOW
        G_PL_lag = h(p, t - T_INTEGRALWINDOW)[2]
    else
        G_PL_lag = G_b    # mirrors: G_PL_lowerbound = G_PL_saved(1) = G_b
    end

    # -- Glucose appearance from meal (Weibull input function) -------------
    # G_meal = σ·k1^σ · t^(σ−1) · exp(−(k1·t)^σ) · D_meal_G
    # At t=0: t^(σ−1) = 0^0.4 = 0.0  →  G_meal = 0  (no singularity)
    G_meal = (t > 0.0) ?
             sigma_p * (k1^sigma_p) * (t^(sigma_p - 1)) *
             exp(-(k1 * t)^sigma_p) * D_meal_G :
             0.0

    # -- Gut glucose mass (EDES_ODE.m line: D.M_G_gut) --------------------
    dM_G_gut = G_meal - k2 * M_G_gut

    # -- Net hepatic glucose flux ------------------------------------------
    G_liv = G_liv_b - k4 * F_I * I_d1 - k3 * (G_PL - G_b)

    # -- Gut → plasma glucose contribution ---------------------------------
    G_gut = k2 * (F_G / (V_G * BW)) * M_G_gut

    # -- Insulin-independent utilisation (brain, erythrocytes) ------------
    U_ii  = G_liv_b * ((KM_p + G_b) / G_b) * (G_PL / (KM_p + G_PL))

    # -- Insulin-dependent utilisation (liver, muscle, adipose) -----------
    U_id  = k5 * F_I * I_d1 * (G_PL / (KM_p + G_PL))

    # -- Renal glucose extraction -----------------------------------------
    U_ren = (G_PL > G_TH_PL) ?
            (C1 / (V_G * BW)) * (G_PL - G_TH_PL) : 0.0

    # -- Plasma glucose rate of change  (EDES_ODE.m: D.G_PL) --------------
    dG_PL = G_liv + G_gut - U_ii - U_id - U_ren

    # -- Moving-window integral of G_PL  (EDES_ODE.m: D.G_int) -----------
    # dG_int/dt = (G_PL − G_b) − (G_PL(t−30) − G_b) = G_PL − G_PL(t−30)
    dG_int = (G_PL - G_b) - (G_PL_lag - G_b)

    # -- Pancreatic insulin production (PID controller) -------------------
    # I_pnc = (1/f_I) * [k6*(G−G_b)  +  (k7/τ_i)*G_int  +
    #                     (k7/τ_i)*G_b  +  k8*τ_d * dG_PL/dt ]
    # Note: dG_PL is computed above → no circular dependency
    I_pnc = (1.0 / F_I) * (k6 * (G_PL - G_b) +
                             (k7 / TAU_I) * G_int +
                             (k7 / TAU_I) * G_b   +
                             (k8 * TAU_D) * dG_PL)

    # -- Hepatic insulin clearance  (EDES_ODE.m: I_liv) -------------------
    I_liv = k7 * (G_b / (F_I * TAU_I * I_PL_b)) * I_PL

    # -- Insulin transport plasma → remote compartment --------------------
    i_rem = k9 * (I_PL - I_PL_b)

    # -- Plasma insulin rate of change  (EDES_ODE.m: D.I_PL) -------------
    dI_PL = I_pnc - I_liv - i_rem

    # -- Remote / interstitial insulin  (EDES_ODE.m: D.I_d1) -------------
    dI_d1 = k9 * (I_PL - I_PL_b) - k10 * I_d1

    # -- Write derivatives back -------------------------------------------
    du[1] = dM_G_gut
    du[2] = dG_PL
    du[3] = dG_int
    du[4] = dI_PL
    du[5] = dI_d1
end

# ============================================================================
# SECTION 6 — RUN A SINGLE EDES SIMULATION
# Returns (G_sim, I_sim) vectors on tgrid, or nothing on failure.
# (mirrors the ode45 call in Generate_VirtualPopulation.m lines 154-167)
# ============================================================================
function run_edes(params::Vector{Float64}, BW::Float64, D_meal_G::Float64,
                  tgrid::AbstractRange{Float64})

    G_b    = params[13]
    I_PL_b = params[14]

    # Initial conditions [M_G_gut, G_PL, G_int, I_PL, I_d1]
    # (mirrors EDES_Initial.m: x0 = [0, G_b, 0, I_PL_b, 0])
    u0 = [0.0, G_b, 0.0, I_PL_b, 0.0]

    # History function for t < t0 = 0:
    # Returns the constant fasting initial state.
    # Replaces the MATLAB globals t_saved=0, G_PL_saved=G_b at t=0.
    h_fun(p_inner, t_lag) = u0

    p     = (params, BW, D_meal_G)
    tspan = (Float64(first(tgrid)), Float64(last(tgrid)))

    prob = DDEProblem(edes_dde!, u0, h_fun, tspan, p;
                      constant_lags = [T_INTEGRALWINDOW])

    sol = try
        solve(prob, MethodOfSteps(Tsit5());
              reltol  = 1e-5,
              abstol  = 1e-8,
              saveat  = collect(tgrid))
    catch
        return nothing   # mirrors: catch → continue in MATLAB loop
    end

    # Check solver succeeded (mirrors: if length(T) ~= n_t → continue)
    if sol.retcode != ReturnCode.Success || length(sol.t) != length(tgrid)
        return nothing
    end

    # Extract plasma glucose (state 2) and insulin (state 4)
    G_sim = [sol.u[i][2] for i in eachindex(sol.t)]   # mirrors X(:,2)
    I_sim = [sol.u[i][4] for i in eachindex(sol.t)]   # mirrors X(:,4)

    return G_sim, I_sim
end

# ============================================================================
# SECTION 7 — VIRTUAL POPULATION GENERATION
# Direct translation of Generate_VirtualPopulation.m main loop.
# ============================================================================
function generate_virtual_population(;
        N_sim::Int           = N,
        seed::Int            = SEED,
        tgrid::AbstractRange = TIME_SIM,
        meal_g::Float64      = MEAL_G)

    rng  = MersenneTwister(seed)   # mirrors rng(42)
    n_t  = length(tgrid)
    n_lhs = size(PARAM_BOUNDS, 1)  # = 7

    # ── Latin Hypercube Sampling (mirrors lhsdesign + scaling) ───────────
    println("Generating LHS design ($N_sim × $n_lhs)...")
    lhs_unit   = lhsdesign(N_sim, n_lhs; rng = rng)
    lb         = PARAM_BOUNDS[:, 1]
    ub         = PARAM_BOUNDS[:, 2]
    # lhs_scaled = lhs_unit .* (ub - lb)' + lb'
    lhs_scaled = lhs_unit .* (ub .- lb)' .+ lb'

    # ── Pre-allocate (NaN = not yet accepted) ────────────────────────────
    glucose_clean = fill(NaN, N_sim, n_t)
    insulin_clean = fill(NaN, N_sim, n_t)
    glucose_noisy = fill(NaN, N_sim, n_t)
    insulin_noisy = fill(NaN, N_sim, n_t)
    param_matrix  = fill(NaN, N_sim, 7)
    valid_flag    = falses(N_sim)

    println("Starting virtual population generation (N=$N_sim)...")

    for i in 1:N_sim

        # Progress display (mirrors mod(i,100)==0 block)
        if mod(i, 100) == 0
            n_valid = sum(valid_flag)
            println("  Individual $i/$N_sim | Valid so far: $n_valid")
        end

        # ── Unpack sampled parameters ─────────────────────────────────
        k1     = lhs_scaled[i, 1]
        k5     = lhs_scaled[i, 2]
        k6     = lhs_scaled[i, 3]
        k8     = lhs_scaled[i, 4]
        G_b    = lhs_scaled[i, 5]
        I_PL_b = lhs_scaled[i, 6]
        BW     = lhs_scaled[i, 7]

        # ── Build full 15-element parameter vector ────────────────────
        # (mirrors Generate_VirtualPopulation.m lines 111-126)
        params = [k1, K2, K3, K4, k5, k6, K7, k8,
                  K9, K10, SIGMA, KM, G_b, I_PL_b, G_LIV_B]

        # ── Run ODE/DDE solver ────────────────────────────────────────
        result = run_edes(params, BW, meal_g, tgrid)
        result === nothing && continue

        G_sim, I_sim = result

        # ── Quality control filters ───────────────────────────────────
        # 1) No negative values  (mirrors lines 174-176)
        (any(G_sim .< 0.0) || any(I_sim .< 0.0)) && continue

        # 2) Glucose within physiological range  (mirrors lines 179-181)
        (maximum(G_sim) > 30.0 || minimum(G_sim) < 2.0) && continue

        # 3) Insulin within physiological range  (mirrors lines 184-186)
        (maximum(I_sim) > 200.0 || minimum(I_sim) < 0.0) && continue

        # ── Add realistic measurement noise ──────────────────────────
        # (mirrors Generate_VirtualPopulation.m lines 200-208)
        noise_G_pct = 0.02 + 0.01 * rand(rng)        # 2–3% uniform
        noise_I_pct = 0.05 + 0.03 * rand(rng)        # 5–8% uniform

        G_noisy_i = G_sim .* (1.0 .+ noise_G_pct .* randn(rng, n_t))
        I_noisy_i = I_sim .* (1.0 .+ noise_I_pct .* randn(rng, n_t))

        G_noisy_i = max.(G_noisy_i, 0.0)   # clamp to non-negative
        I_noisy_i = max.(I_noisy_i, 0.0)

        # ── Store accepted individual ─────────────────────────────────
        glucose_clean[i, :] = G_sim
        insulin_clean[i, :] = I_sim
        glucose_noisy[i, :] = G_noisy_i
        insulin_noisy[i, :] = I_noisy_i
        param_matrix[i, :]  = [k1, k5, k6, k8, G_b, I_PL_b, BW]
        valid_flag[i]        = true
    end

    # ── Trim to accepted individuals (mirrors lines 223-232) ─────────────
    idx_valid = findall(valid_flag)
    n_valid   = length(idx_valid)
    @printf("\nSimulation complete. Accepted: %d / %d individuals (%.1f%%)\n",
            n_valid, N_sim, 100.0 * n_valid / N_sim)

    # Return as NamedTuple (mirrors virtual_population struct in MATLAB)
    vp = (
        time          = collect(tgrid),          # [n_t]  (row vector in MATLAB)
        glucose_clean = glucose_clean[idx_valid, :],  # [n_valid × n_t]
        insulin_clean = insulin_clean[idx_valid, :],
        glucose_noisy = glucose_noisy[idx_valid, :],
        insulin_noisy = insulin_noisy[idx_valid, :],
        param_matrix  = param_matrix[idx_valid, :],   # [n_valid × 7]
        param_names   = ["k1", "k5", "k6", "k8", "G_b", "I_PL_b", "BW"],
        n_valid       = n_valid,
        N_attempted   = N_sim,
    )

    return vp
end

# ============================================================================
# SECTION 8 — ADA LABELLING
# Direct translation of Label_VirtualPopulation.m.
#
# ADA criteria (applied to glucose_noisy):
#   T2DM : fasting >= 7.0  AND  2-h >= 11.1       (line 51 in current .m)
#   IGT  : NOT T2DM  AND  7.8 <= 2-h <= 11.1      (line 54)
#   NGT  : fasting < 5.6  AND  2-h < 7.8          (line 55)
#
# Labels default to "NGT", overwritten with "IGT" then "T2DM"
# (mirrors lines 57-59: repmat / labels(is_IGT) / labels(is_T2DM))
# ============================================================================
function label_virtual_population(vp::NamedTuple)

    time_vec      = vp.time
    glucose_noisy = vp.glucose_noisy
    n_valid       = vp.n_valid

    # ── Locate t=0 and t=120 column indices ──────────────────────────────
    # (mirrors lines 35-41: find(time==0) / find(time==120))
    idx_fast = findfirst(==(0.0),   time_vec)
    idx_2h   = findfirst(==(120.0), time_vec)

    (idx_fast === nothing || idx_2h === nothing) &&
        error("Cannot locate t=0 or t=120 in the time vector.")

    G_fasting = glucose_noisy[:, idx_fast]   # [n_valid]   mirrors line 43
    G_2h      = glucose_noisy[:, idx_2h]     # [n_valid]   mirrors line 44

    # ── Apply ADA rules ───────────────────────────────────────────────────
    is_T2DM = (G_fasting .>= 7.0) .& (G_2h .>= 11.1)            # line 51
    is_IGT  = .!is_T2DM .& (G_2h .>= 7.8) .& (G_2h .<= 11.1)   # line 54
    is_NGT  = (G_fasting .< 5.6)  .& (G_2h .< 7.8)              # line 55

    # Default label = "NGT", then overwrite (mirrors lines 57-59)
    labels = fill("NGT", n_valid)
    labels[is_IGT]  .= "IGT"
    labels[is_T2DM] .= "T2DM"

    n_NGT  = sum(is_NGT)
    n_IGT  = sum(is_IGT)
    n_T2DM = sum(is_T2DM)

    @printf("\nADA labelling results (N = %d):\n", n_valid)
    @printf("  NGT  : %4d  (%.1f%%)\n", n_NGT,  100.0 * n_NGT  / n_valid)
    @printf("  IGT  : %4d  (%.1f%%)\n", n_IGT,  100.0 * n_IGT  / n_valid)
    @printf("  T2DM : %4d  (%.1f%%)\n", n_T2DM, 100.0 * n_T2DM / n_valid)

    # ── Build per-category datasets (mirrors make_subset anonymous fn) ────
    function make_subset(mask::BitVector, cat_name::String)
        return (
            category      = cat_name,
            time          = vp.time,
            glucose_noisy = vp.glucose_noisy[mask, :],
            insulin_noisy = vp.insulin_noisy[mask, :],
            glucose_clean = vp.glucose_clean[mask, :],
            insulin_clean = vp.insulin_clean[mask, :],
            param_matrix  = vp.param_matrix[mask, :],
            param_names   = vp.param_names,
            G_fasting     = G_fasting[mask],
            G_2h          = G_2h[mask],
            labels        = labels[mask],
            n             = sum(mask),
        )
    end

    dataset_NGT  = make_subset(is_NGT,  "NGT")
    dataset_IGT  = make_subset(is_IGT,  "IGT")
    dataset_T2DM = make_subset(is_T2DM, "T2DM")

    # ── Annotate the full population struct (mirrors lines 94-99) ────────
    vp_labelled = merge(vp, (
        labels    = labels,
        G_fasting = G_fasting,
        G_2h      = G_2h,
        is_NGT    = is_NGT,
        is_IGT    = is_IGT,
        is_T2DM   = is_T2DM,
    ))

    return vp_labelled, dataset_NGT, dataset_IGT, dataset_T2DM
end

# ============================================================================
# SECTION 9 — ENTRY POINT
# Mirrors the sequential execution of the two MATLAB scripts.
# ============================================================================
function main()

    # ── Step 1: Generate virtual population ──────────────────────────────
    vp = generate_virtual_population()

    # ── Step 2: Save raw population (mirrors save('virtual_population.mat'))
    println("\nSaving virtual_population.jld2...")
    jldsave("virtual_population.jld2"; virtual_population = vp)
    println("  Saved: virtual_population.jld2")

    # ── Step 3: ADA labelling ─────────────────────────────────────────────
    vp_labelled, dataset_NGT, dataset_IGT, dataset_T2DM =
        label_virtual_population(vp)

    # ── Step 4: Save labelled population ──────────────────────────────────
    # (mirrors save('virtual_population_labelled.mat', ...))
    println("\nSaving virtual_population_labelled.jld2...")
    jldsave("virtual_population_labelled.jld2";
            virtual_population = vp_labelled,
            dataset_NGT        = dataset_NGT,
            dataset_IGT        = dataset_IGT,
            dataset_T2DM       = dataset_T2DM)
    println("  Saved: virtual_population_labelled.jld2")
    println("  Fields: virtual_population, dataset_NGT, dataset_IGT, dataset_T2DM")

    return vp_labelled, dataset_NGT, dataset_IGT, dataset_T2DM
end

# ── Run automatically when called as a script ─────────────────────────────
# julia GenerateAndLabel_VirtualPopulation.jl
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
