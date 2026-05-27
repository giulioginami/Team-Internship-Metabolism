%% Save_PID_Predictions.m
% For every patient in all three populations, runs the EDES ODE three times
% (once per optimised PID expert: NGT, IGT, T2DM) and saves the predicted
% glucose and insulin at the sparse time points to pid_predictions.mat.
%
% Output  pid_predictions.mat
%   G_pids  [N x 3 x 5]  glucose at sparse times  (expert order: NGT, IGT, T2DM)
%   I_pids  [N x 3 x 5]  insulin at sparse times
%   labels  [N x 1]      ground-truth class  1=NGT  2=IGT  3=T2DM

clear; clc;
global t_saved G_PL_saved

%% -------------------------------------------------------------------------
%% Optimised PID parameters 
%% -------------------------------------------------------------------------
%          k5       k6       k8
pids = [  0.092,   0.079,   7.394;   % NGT  expert 
          0.006,   0.089,   4.724;   % IGT  expert
          0.014,   0.000,   5.755];  % T2DM expert 

%% -------------------------------------------------------------------------
%% Fixed parameters (Rozendaal et al. 2018)
%% -------------------------------------------------------------------------
fixed.k2      = 0.28;
fixed.k3      = 6.07e-3;
fixed.k4      = 2.35e-4;
fixed.k7      = 1.15;
fixed.k9      = 3.83e-2;
fixed.k10     = 2.84e-1;
fixed.sigma   = 1.4;
fixed.KM      = 13.2;
fixed.G_liv_b = 0.043;

%% -------------------------------------------------------------------------
%% Load sparse data
%% -------------------------------------------------------------------------
fprintf('Loading virtual_population_sparse.mat...\n');
load('virtual_population_sparse.mat', ...
     't_sparse', 'dataset_NGT_sparse', 'dataset_IGT_sparse', 'dataset_T2DM_sparse');

meal_G    = 75000;
time_full = (0:1:480)';

% Indices of sparse time points in the full time vector
sparse_idx = arrayfun(@(t) find(time_full == t, 1), t_sparse);
n_sparse   = numel(t_sparse);

%% -------------------------------------------------------------------------
%% Stack all patients from the three populations
%% -------------------------------------------------------------------------
datasets = {dataset_NGT_sparse, dataset_IGT_sparse, dataset_T2DM_sparse};
n_pops   = [dataset_NGT_sparse.n, dataset_IGT_sparse.n, dataset_T2DM_sparse.n];
N        = sum(n_pops);

% Pre-allocate
G_pids = NaN(N, 3, n_sparse);   % [N x 3 experts x 5 time points]
I_pids = NaN(N, 3, n_sparse);
labels = NaN(N, 1);

fprintf('Total patients: %d  (NGT=%d  IGT=%d  T2DM=%d)\n', ...
        N, n_pops(1), n_pops(2), n_pops(3));

%% -------------------------------------------------------------------------
%% Main loop: patients × PID experts
%% -------------------------------------------------------------------------
patient_offset = 0;

for pop = 1:3

    ds    = datasets{pop};
    n_pat = ds.n;

    fprintf('\n--- Population %d/3  (%s, n=%d) ---\n', pop, ds.category, n_pat);

    for i = 1:n_pat

        if mod(i, 50) == 0
            fprintf('  Patient %d / %d\n', i, n_pat);
        end

        % Patient-specific parameters (from param_matrix columns: k1,k5,k6,k8,G_b,I_PL_b,BW)
        k1     = ds.param_matrix(i, 1);
        G_b    = ds.param_matrix(i, 5);
        I_PL_b = ds.param_matrix(i, 6);
        BW     = ds.param_matrix(i, 7);

        global_idx = patient_offset + i;
        labels(global_idx) = pop;   % 1=NGT  2=IGT  3=T2DM

        % Run one simulation per PID expert
        for e = 1:3
            k5 = pids(e, 1);
            k6 = pids(e, 2);
            k8 = pids(e, 3);

            [G_sim, I_sim] = run_simulation( ...
                k5, k6, k8, k1, time_full, fixed, G_b, I_PL_b, BW, meal_G);

            G_pids(global_idx, e, :) = G_sim(sparse_idx);
            I_pids(global_idx, e, :) = I_sim(sparse_idx);
        end
    end

    patient_offset = patient_offset + n_pat;
end

%% -------------------------------------------------------------------------
%% Save
%% -------------------------------------------------------------------------
fprintf('\nSaving pid_predictions.mat...\n');
save('EDES_MoE/PID_Optimization/pid_predictions.mat', 'G_pids', 'I_pids', 'labels', 't_sparse', '-v7.3');
fprintf('Done.  G_pids: [%s]  I_pids: [%s]  labels: [%s]\n', ...
        num2str(size(G_pids)), num2str(size(I_pids)), num2str(size(labels)));

%% -------------------------------------------------------------------------
%% Helper: run one EDES simulation, return full glucose and insulin traces
%% -------------------------------------------------------------------------
function [G_sim, I_sim] = run_simulation(k5, k6, k8, k1, time, fixed, G_b, I_PL_b, BW, meal_G)
    global t_saved G_PL_saved

    parameters      = zeros(1, 15);
    parameters(1)   = k1;
    parameters(2)   = fixed.k2;
    parameters(3)   = fixed.k3;
    parameters(4)   = fixed.k4;
    parameters(5)   = k5;
    parameters(6)   = k6;
    parameters(7)   = fixed.k7;
    parameters(8)   = k8;
    parameters(9)   = fixed.k9;
    parameters(10)  = fixed.k10;
    parameters(11)  = fixed.sigma;
    parameters(12)  = fixed.KM;
    parameters(13)  = G_b;
    parameters(14)  = I_PL_b;
    parameters(15)  = fixed.G_liv_b;

    c.f_G              = 0.005551;
    c.f_I              = 1;
    c.V_G              = 17/70;
    c.G_liv_b          = fixed.G_liv_b;
    c.tau_i            = 31;
    c.tau_d            = 3;
    c.G_th_PL          = 9;
    c.t_integralwindow = 30;
    c.c1               = 0.1;

    input_data.glucose = G_b;
    input_data.insulin = I_PL_b;
    input_data.BW      = BW;
    input_data.meal.G  = meal_G;

    x0 = [0, G_b, 0, I_PL_b, 0];

    t_saved    = 0;
    G_PL_saved = G_b;

    ODE_options = odeset('RelTol', 1e-5, 'AbsTol', 1e-8, 'OutputFcn', @integratorfunG);

    try
        [~, X] = ode45(@EDES_ODE, time, x0, ODE_options, parameters, c, input_data, 1);
        if size(X, 1) == length(time)
            G_sim = X(:, 2);
            I_sim = X(:, 4);
        else
            G_sim = G_b    * ones(length(time), 1);
            I_sim = I_PL_b * ones(length(time), 1);
        end
    catch
        G_sim = G_b    * ones(length(time), 1);
        I_sim = I_PL_b * ones(length(time), 1);
    end
end
