%% final_architecture.m
% Full Mixture-of-Experts personalised parameter estimation.
%
% Given sparse OGTT observations for one patient:
%   1. Compute gating weights via the trained gating network (pure MATLAB)
%   2. Optimise patient-specific k1 and k5 using the weighted EDES model
%   3. Plot: 3 expert curves (dashed) + MoE weighted (solid) + observed data
%
% Prerequisites:
%   - virtual_population_sparse.mat   (Create_SparseDatasets.m)
%   - gating_weights.mat              (gating_network.py, run after training)
%   - EDES_ODE.m, integratorfunG.m

clear; clc;
global t_saved G_PL_saved

%% -------------------------------------------------------------------------
%% Settings
%% -------------------------------------------------------------------------
POPULATION  = 'T2DM';   % 'NGT', 'IGT', or 'T2DM'
PATIENT_IDX = 50;       % index within the selected population

%% -------------------------------------------------------------------------
%% Optimised PID expert parameters  [k5, k6, k8]
%% -------------------------------------------------------------------------
pids = [  0.092,   0.079,   7.394;   % NGT  expert
          0.006,   0.089,   4.724;   % IGT  expert
          0.014,   0.000,   5.755];  % T2DM expert

expert_names = {'NGT', 'IGT', 'T2DM'};
colors       = {[0.18 0.63 0.18], [0.93 0.69 0.13], [0.80 0.15 0.15]};

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
%% Load sparse data and select patient
%% -------------------------------------------------------------------------
fprintf('Loading virtual_population_sparse.mat...\n');
load('virtual_population_sparse.mat', 't_sparse', ...
     'dataset_NGT_sparse', 'dataset_IGT_sparse', 'dataset_T2DM_sparse');

switch POPULATION
    case 'NGT',  ds = dataset_NGT_sparse;
    case 'IGT',  ds = dataset_IGT_sparse;
    case 'T2DM', ds = dataset_T2DM_sparse;
    otherwise,   error('Unknown population: %s', POPULATION);
end

G_obs = ds.glucose_noisy(PATIENT_IDX, :);   % [1 x 5]
I_obs = ds.insulin_noisy(PATIENT_IDX, :);   % [1 x 5]

% param_matrix columns: k1(1), k5(2), k6(3), k8(4), G_b(5), I_PL_b(6), BW(7)
k1_true = ds.param_matrix(PATIENT_IDX, 1);
G_b     = ds.param_matrix(PATIENT_IDX, 5);
I_PL_b  = ds.param_matrix(PATIENT_IDX, 6);
BW      = ds.param_matrix(PATIENT_IDX, 7);

fprintf('Patient %d  |  population: %s  |  G_b=%.2f  I_PL_b=%.2f  BW=%.1f  k1_true=%.4f\n', ...
    PATIENT_IDX, POPULATION, G_b, I_PL_b, BW, k1_true);

%% -------------------------------------------------------------------------
%% Load patient data from sample_data.mat
%% -------------------------------------------------------------------------
% PATIENT_IDX = 1;       % 1 to 4
% POPULATION  = 'real';  % label used in plot titles
% 
% fprintf('Loading sample_data.mat...\n');
% raw = load('sample_data.mat');
% sd  = raw.sample_data;
% 
% % sample_data has 6 time points: [0, 60, 120, 240, 360, 480] min
% G_raw = sd.glucose(PATIENT_IDX, :);   % [1 x 6]
% I_raw = sd.insulin(PATIENT_IDX, :);   % [1 x 6]
% BW    = sd.BW(PATIENT_IDX);
% 
% G_b    = G_raw(1);   % fasting glucose  (t = 0)
% I_PL_b = I_raw(1);   % fasting insulin  (t = 0)
% 
% % Gating network expects 7 time points: [0,30,60,120,240,360,480]
% % t=30 is missing → interpolate as midpoint of t=0 and t=60
% G_30 = (G_raw(1) + G_raw(2)) / 2;
% I_30 = (I_raw(1) + I_raw(2)) / 2;
% 
% t_sparse = [0, 30, 60, 120, 240, 360, 480];
% G_obs    = [G_raw(1), G_30, G_raw(2:end)];   % [1 x 7]
% I_obs    = [I_raw(1), I_30, I_raw(2:end)];   % [1 x 7]
% 
% % k1 is not available in sample_data — use population median as warm start
% k1_true = 0.028;
% 
% fprintf('Patient %d  |  G_b=%.2f  I_PL_b=%.2f  BW=%.1f  k1_init=%.4f\n', ...
%     PATIENT_IDX, G_b, I_PL_b, BW, k1_true);

%% -------------------------------------------------------------------------
%% Load gating network weights
%% -------------------------------------------------------------------------
fprintf('Loading gating_weights.mat...\n');
gw = load('gating_weights.mat');
W1 = gw.W1;        % [32 x 10]
b1 = gw.b1(:);     % [32 x 1]
W2 = gw.W2;        % [32 x 32]
b2 = gw.b2(:);     % [32 x 1]
W3 = gw.W3;        % [3  x 32]
b3 = gw.b3(:);     % [3  x 1]
X_mean = gw.X_mean(:)';   % [1 x 10]
X_std  = gw.X_std(:)';    % [1 x 10]

%% -------------------------------------------------------------------------
%% Gating network forward pass  (Linear → ReLU → Linear → ReLU → Linear → Softmax)
%% -------------------------------------------------------------------------
x_in   = [G_obs, I_obs];              % [1 x 10]
x_norm = (x_in - X_mean) ./ X_std;   % [1 x 10]

h1 = max(0, W1 * x_norm' + b1);      % [32 x 1]
h2 = max(0, W2 * h1      + b2);      % [32 x 1]
z  = W3 * h2 + b3;                   % [3  x 1]
e_z = exp(z - max(z));               % numerically stable softmax
w   = double(e_z / sum(e_z));        % [3  x 1]  gating weights  (cast to double)

fprintf('\nGating weights:  w_NGT=%.3f  w_IGT=%.3f  w_T2DM=%.3f\n', w(1), w(2), w(3));

%% -------------------------------------------------------------------------
%% Optimisation: k1 and k5
%% -------------------------------------------------------------------------
meal_G    = 75000;
time_full = (0:1:240)';

sparse_idx = arrayfun(@(t) find(time_full == t, 1), t_sparse);

% Initial guesses
k5_init = w' * pids(:, 1);   % weighted average of expert k5 values
k1_init = k1_true;            % use patient's own k1 as warm start

k0 = double([k1_init, k5_init]);
lb = [0.01,  0.0 ];
ub = [0.05,  0.17];

fprintf('Optimising [k1, k5]  (init: k1=%.4f  k5=%.4f) ...\n', k0(1), k0(2));

opt_options = optimoptions('lsqnonlin', 'Display', 'iter', ...
    'MaxFunctionEvaluations', 400, 'FunctionTolerance', 1e-8);

[k_opt, resnorm] = lsqnonlin( ...
    @(k) weighted_residuals(k, w, pids, fixed, G_b, I_PL_b, BW, meal_G, ...
                            time_full, sparse_idx, G_obs, I_obs), ...
    k0, lb, ub, opt_options);

fprintf('\nOptimised  k1=%.4f  k5=%.4f  resnorm=%.6f\n', k_opt(1), k_opt(2), resnorm);

%% -------------------------------------------------------------------------
%% Final simulations with optimised parameters
%% -------------------------------------------------------------------------
k1_opt = k_opt(1);
k5_opt = k_opt(2);

G_experts = zeros(3, length(time_full));
I_experts = zeros(3, length(time_full));

for e = 1:3
    k6 = pids(e, 2);
    k8 = pids(e, 3);
    [G_sim, I_sim] = run_simulation(k1_opt, k5_opt, k6, k8, time_full, fixed, G_b, I_PL_b, BW, meal_G);
    G_experts(e, :) = G_sim';
    I_experts(e, :) = I_sim';
end

% MoE weighted prediction
G_moe = w(1)*G_experts(1,:) + w(2)*G_experts(2,:) + w(3)*G_experts(3,:);
I_moe = w(1)*I_experts(1,:) + w(2)*I_experts(2,:) + w(3)*I_experts(3,:);

%% -------------------------------------------------------------------------
%% Plot
%% -------------------------------------------------------------------------
figure('Name', 'MoE Personalised Parameter Estimation', 'Position', [80 80 1100 460]);

subplot(1, 2, 1);
hold on;
for e = 1:3
    plot(time_full, G_experts(e, :), '--', 'Color', colors{e}, 'LineWidth', 1.4, ...
         'DisplayName', [expert_names{e} ' expert']);
end
plot(time_full, G_moe, 'k-',  'LineWidth', 2.4, 'DisplayName', 'MoE weighted');
plot(t_sparse,  G_obs, 'ko',  'MarkerSize', 7, 'MarkerFaceColor', 'k', ...
     'DisplayName', 'Observed');
xlabel('Time (min)'); ylabel('Glucose (mmol/L)');
title(sprintf('Glucose — %s patient %d\nk1=%.4f  k5=%.4f', ...
    POPULATION, PATIENT_IDX, k1_opt, k5_opt));
legend('Location', 'best'); grid on;

subplot(1, 2, 2);
hold on;
for e = 1:3
    plot(time_full, I_experts(e, :), '--', 'Color', colors{e}, 'LineWidth', 1.4, ...
         'DisplayName', [expert_names{e} ' expert']);
end
plot(time_full, I_moe, 'k-',  'LineWidth', 2.4, 'DisplayName', 'MoE weighted');
plot(t_sparse,  I_obs, 'ko',  'MarkerSize', 7, 'MarkerFaceColor', 'k', ...
     'DisplayName', 'Observed');
xlabel('Time (min)'); ylabel('Insulin (mU/L)');
title(sprintf('Insulin — %s patient %d', POPULATION, PATIENT_IDX));
legend('Location', 'best'); grid on;

sgtitle(sprintf('MoE personalised fit  |  w = [%.2f  %.2f  %.2f]  (NGT / IGT / T2DM)', ...
    w(1), w(2), w(3)));

%% =========================================================================
%% Local functions
%% =========================================================================

function res = weighted_residuals(k, w, pids, fixed, G_b, I_PL_b, BW, meal_G, ...
                                  time_full, sparse_idx, G_obs, I_obs)
    k1 = k(1);
    k5 = k(2);

    G_pred = zeros(1, numel(sparse_idx));
    I_pred = zeros(1, numel(sparse_idx));

    for e = 1:3
        k6 = pids(e, 2);
        k8 = pids(e, 3);
        [G_sim, I_sim] = run_simulation(k1, k5, k6, k8, time_full, fixed, G_b, I_PL_b, BW, meal_G);
        G_pred = G_pred + w(e) * G_sim(sparse_idx)';
        I_pred = I_pred + w(e) * I_sim(sparse_idx)';
    end

    res = [(G_pred - G_obs) / norm(G_obs), ...
           (I_pred - I_obs) / norm(I_obs)];
end

function [G_sim, I_sim] = run_simulation(k1, k5, k6, k8, time, fixed, G_b, I_PL_b, BW, meal_G)
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
