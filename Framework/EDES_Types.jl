# EDES_Types.jl

# --- Controller Interface ---
abstract type AbstractMetabolicController end

struct PIDController <: AbstractMetabolicController
    k6::Float64
    k7::Float64
    k8::Float64
end

struct MoEController{C1, C2, C3} <: AbstractMetabolicController
    expert_healthy::C1
    expert_igt::C2
    expert_t2d::C3
    weights::Vector{Float64}
end

#struct NeuralNetController <: AbstractMetabolicController
    #model::Any # Placeholder for a Flux Chain
#end

# --- System Parameters ---
Base.@kwdef struct EDESConstants
    f_G::Float64              = 0.005551
    f_I::Float64              = 1.0      
    V_G::Float64              = 17/70    
    tau_i::Float64            = 31.0     
    tau_d::Float64            = 3.0      
    G_th_PL::Float64          = 9.0      
    c1::Float64               = 0.1      
    t_integralwindow::Float64 = 120.0  
    N_delay::Int              = 3        
end

Base.@kwdef struct EDESInputs
    D_meal_G::Float64 = 75000.0
    BW::Float64       = 75.0
end

Base.@kwdef struct EDESParameters
    k1::Float64 = 0.02
    k2::Float64 = 0.28
    k3::Float64 = 6.07e-3
    k4::Float64 = 2.35e-4
    k5::Float64 = 0.05
    k9::Float64 = 3.83e-2
    k10::Float64= 2.84e-1
    sigma::Float64   = 1.4
    KM::Float64      = 13.2
    G_b::Float64     = 5.0
    I_PL_b::Float64  = 5.0
    G_liv_b::Float64 = 0.043
end