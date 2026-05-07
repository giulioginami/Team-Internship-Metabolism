# EDES_Core.jl

# --- Controller Implementations ---
function compute_control(ctrl::PIDController, G_PL, G_int, dG_PL, I_PL, params, constants)
    I_pnc = (1.0 / constants.f_I) * (ctrl.k6 * (G_PL - params.G_b) + 
            (ctrl.k7 / constants.tau_i) * G_int + 
            (ctrl.k7 / constants.tau_i) * params.G_b + 
            (ctrl.k8 * constants.tau_d) * dG_PL)
            
    I_liv = ctrl.k7 * (params.G_b / (constants.f_I * constants.tau_i * params.I_PL_b)) * I_PL
    
    return I_pnc, I_liv
end

# --- New MoE Dispatch ---
function compute_control(ctrl::MoEController, G_PL, G_int, dG_PL, I_PL, params, constants)
    # Get control outputs from each expert
    pnc_1, liv_1 = compute_control(ctrl.expert_healthy, G_PL, G_int, dG_PL, I_PL, params, constants)
    pnc_2, liv_2 = compute_control(ctrl.expert_igt, G_PL, G_int, dG_PL, I_PL, params, constants)
    pnc_3, liv_3 = compute_control(ctrl.expert_t2d, G_PL, G_int, dG_PL, I_PL, params, constants)
    
    # Extract weights
    w1, w2, w3 = ctrl.weights
    
    # Blend the actions
    I_pnc = (w1 * pnc_1) + (w2 * pnc_2) + (w3 * pnc_3)
    I_liv = (w1 * liv_1) + (w2 * liv_2) + (w3 * liv_3)
    
    return I_pnc, I_liv
end

# --- Main DDE Core ---
function edes_dde_modular!(du, u, h, p, t)
    # Unpack 5 biological states
    M_G_gut, G_PL, G_int, I_PL, I_d1 = u
    
    params, constants, inputs, controller = p
    
    # ─── 1. Meal Appearance ───
    t_safe = max(t, 1e-6) 
    G_meal = params.sigma * (params.k1^params.sigma) * (t_safe^(params.sigma - 1)) * exp(-(params.k1 * t_safe)^params.sigma) * inputs.D_meal_G
    du[1] = G_meal - params.k2 * M_G_gut

    # ─── 2. Plasma Glucose ───
    G_liv = params.G_liv_b - params.k4 * constants.f_I * I_d1 - params.k3 * (G_PL - params.G_b)
    G_gut = params.k2 * (constants.f_G / (constants.V_G * inputs.BW)) * M_G_gut
    
    U_ii  = params.G_liv_b * ((params.KM + params.G_b) / params.G_b) * (G_PL / (params.KM + G_PL))
    U_id  = params.k5 * constants.f_I * I_d1 * (G_PL / (params.KM + G_PL))
    U_ren = (G_PL > constants.G_th_PL) ? (constants.c1 / (constants.V_G * inputs.BW) * (G_PL - constants.G_th_PL)) : 0.0
    
    # Derivative of Plasma Glucose
    du[2] = G_liv + G_gut - U_ii - U_id - U_ren

    # ─── 3. Sliding Window Integral (Exact Delay) ───
    if t > constants.t_integralwindow
        # Look exactly 120 minutes into the past using the history function `h`
        delayed_state = h(p, t - constants.t_integralwindow)
        G_PL_lowerbound = delayed_state[2] # Index 2 is G_PL
    else
        G_PL_lowerbound = params.G_b
    end
    
    du[3] = (G_PL - params.G_b) - (G_PL_lowerbound - params.G_b)

    # ─── 4. Modular Control Action ───
    # Note we pass du[2] directly as it is the current rate of change of glucose
    I_pnc, I_liv = compute_control(controller, G_PL, G_int, du[2], I_PL, params, constants)

    
    # ─── 5. Plasma & Remote Insulin ───
    i_rem = params.k9 * (I_PL - params.I_PL_b)
    
    du[4] = I_pnc - I_liv - i_rem
    du[5] = i_rem - params.k10 * I_d1
end

# History function defines the state of the patient BEFORE t = 0
function history_func(p, t)
    params, _, _, _ = p
    # Assume patient has been resting at basal levels prior to meal
    return [0.0, params.G_b, 0.0, params.I_PL_b, 0.0]
end