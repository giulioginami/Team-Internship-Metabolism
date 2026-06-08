%% main.m
% Direct comparison of the MoE and personalized-PID approaches on one patient
% from the Ohashi dataset.
%
%   1. Run MoE optimisation       [k1, k5]        
%   2. Run single-expert optimisation [k1, k5, k6] 
%   3. Figure 1 — Predicted trajectories vs observed (both methods overlaid)
%   4. Figure 2 — RMSE and fitted-parameter comparison
%
% Prerequisites: japan_population_labelled.mat, gating_weights.mat,
%                EDES_ErrorFunc.m, EDES_Parameters.m, EDES_Initial.m,
%                EDES_ODE.m, integratorfunG.m

clear; clc;
global t_saved G_PL_saved

%% -------------------------------------------------------------------------
%% Input
%% -------------------------------------------------------------------------
% To run for a different Ohashi-dataset patient, change PATIENT_IDX (1–118).
%
% To run for a generic patient not in the Ohashi dataset, replace the
% data-loading block below with:
%
%   G_obs      [1 x 5]  plasma glucose at t = [0 30 60 90 120] min  (mmol/L)
%   I_obs      [1 x 5]  plasma insulin at t = [0 30 60 90 120] min  (mU/L)
%   BW                  body weight  (kg)
%   POPULATION          ADA category string: 'NGT', 'IGT', or 'T2DM'
%
% num_par_sets controls the number of LHS starting points for the
% single-expert optimisation (higher = more robust, slower; default: 5).

%% -------------------------------------------------------------------------
%% Settings
%% -------------------------------------------------------------------------
PATIENT_IDX  = 4;   % 1 to 118
num_par_sets = 5;     % LHS starts for single-expert

col_moe = [0.13, 0.47, 0.71];   % blue  — MoE
col_se  = [0.84, 0.37, 0.01];   % orange — single-expert

%% -------------------------------------------------------------------------
%% Load dataset and extract patient
%% -------------------------------------------------------------------------
fprintf('Loading data...\n');
raw = load('japan_population_labelled.mat');
jp  = raw.japan_population;

t_sparse = double(jp.time);   % [0 30 60 90 120]

G_obs  = jp.glucose_noisy(PATIENT_IDX, :);   % [1 x 5]  mmol/L
I_obs  = jp.insulin_noisy(PATIENT_IDX, :);   % [1 x 5]  mU/L
BW     = jp.BW(PATIENT_IDX);
G_b    = G_obs(1);
I_PL_b = I_obs(1);

if      jp.is_NGT(PATIENT_IDX),  POPULATION = 'NGT';
elseif  jp.is_IGT(PATIENT_IDX),  POPULATION = 'IGT';
elseif  jp.is_T2DM(PATIENT_IDX), POPULATION = 'T2DM';
else,                             POPULATION = 'unclassified';
end

fprintf('Patient %d  |  ADA category: %s  |  G_b=%.2f  I_PL_b=%.2f  BW=%.1f\n', ...
    PATIENT_IDX, POPULATION, G_b, I_PL_b, BW);

% Common simulation grid
time_full  = (0:1:240)';
sparse_idx = arrayfun(@(t) find(time_full == t, 1), t_sparse);
meal_G     = 75000;

%% =========================================================================
%% APPROACH 1 — MoE  
%% =========================================================================
fprintf('\n=== Approach 1: MoE ===\n');

pids = [  0.092,   0.079,   7.394;   % NGT
          0.006,   0.089,   4.724;   % IGT
          0.014,   0.000,   5.755];  % T2DM

expert_names = {'NGT', 'IGT', 'T2DM'};

fixed.k2 = 0.28;   fixed.k3 = 6.07e-3;  fixed.k4 = 2.35e-4;
fixed.k7 = 1.15;   fixed.k9 = 3.83e-2;  fixed.k10 = 2.84e-1;
fixed.sigma = 1.4; fixed.KM = 13.2;     fixed.G_liv_b = 0.043;

% Gating network forward pass
gw = load('gating_weights.mat');
W1 = gw.W1;  b1 = gw.b1(:);
W2 = gw.W2;  b2 = gw.b2(:);
W3 = gw.W3;  b3 = gw.b3(:);
X_mean = gw.X_mean(:)';
X_std  = gw.X_std(:)';

x_norm = ([G_obs, I_obs] - X_mean) ./ X_std;
h1  = max(0, W1 * x_norm' + b1);
h2  = max(0, W2 * h1      + b2);
z   = W3 * h2 + b3;
e_z = exp(z - max(z));
w   = double(e_z / sum(e_z));   % [3 x 1]

fprintf('Gating weights:  w_NGT=%.3f  w_IGT=%.3f  w_T2DM=%.3f\n', w(1), w(2), w(3));

% Optimise [k1, k5]
k0_moe = double([0.028, w' * pids(:,1)]);
lb_moe = [0.0,  0.0 ];
ub_moe = [0.05, 0.17];

opt_moe = optimoptions('lsqnonlin', 'Display', 'off', ...
    'MaxFunctionEvaluations', 1000, 'FunctionTolerance', 1e-8);

fprintf('Optimising [k1, k5]...\n');
[k_opt_moe, resnorm_moe] = lsqnonlin( ...
    @(k) weighted_residuals(k, w, pids, fixed, G_b, I_PL_b, BW, meal_G, ...
                            time_full, sparse_idx, G_obs, I_obs), ...
    k0_moe, lb_moe, ub_moe, opt_moe);

fprintf('MoE result:  k1=%.4f  k5=%.4f  resnorm=%.6f\n', ...
    k_opt_moe(1), k_opt_moe(2), resnorm_moe);

% Simulate MoE trajectories
G_experts = zeros(3, length(time_full));
I_experts = zeros(3, length(time_full));
for e = 1:3
    [G_sim, I_sim] = run_simulation(k_opt_moe(1), k_opt_moe(2), pids(e,2), pids(e,3), ...
        time_full, fixed, G_b, I_PL_b, BW, meal_G);
    G_experts(e,:) = G_sim';
    I_experts(e,:) = I_sim';
end
G_moe = (w(1)*G_experts(1,:) + w(2)*G_experts(2,:) + w(3)*G_experts(3,:))';  % [241 x 1]
I_moe = (w(1)*I_experts(1,:) + w(2)*I_experts(2,:) + w(3)*I_experts(3,:))';  % [241 x 1]

%% =========================================================================
%% APPROACH 2 — Single-expert  
%% =========================================================================
fprintf('\n=== Approach 2: Single-expert ===\n');

input_data.glucose = G_obs;
input_data.insulin = I_obs;
input_data.BW      = BW;
input_data.meal.G  = 75000;
input_data.time_G  = t_sparse;
input_data.time_I  = t_sparse;

lb_se = [0.005, 0,  0 ];
ub_se = [0.1,   1,  15];
d_se  = ub_se - lb_se;

time_opt = 0:1:240;   % optimisation span (AUC term uses X(1:240,1))

lsq_se = optimset('Algorithm', 'trust-region-reflective', ...
    'MaxFunEvals', 1000, 'TolX', 1e-8, 'Display', 'off');

lhs_samples  = lhsdesign(num_par_sets, 3);
initial_pars = lhs_samples .* d_se + lb_se;

best_resnorm_se = inf;
best_p_opt_se   = initial_pars(1,:);

fprintf('Optimising [k1, k5, k6] with %d LHS starts...\n', num_par_sets);
for s = 1:num_par_sets
    fprintf('  Start %d/%d  (k1=%.4f  k5=%.4f  k6=%.4f) ...', ...
        s, num_par_sets, initial_pars(s,1), initial_pars(s,2), initial_pars(s,3));
    try
        [p_opt_s, resnorm_s] = lsqnonlin(@EDES_ErrorFunc, initial_pars(s,:), ...
            lb_se, ub_se, lsq_se, input_data, 1, time_opt);
        fprintf('  resnorm = %.6f\n', resnorm_s);
        if resnorm_s < best_resnorm_se
            best_resnorm_se = resnorm_s;
            best_p_opt_se   = p_opt_s;
        end
    catch
        fprintf('  FAILED\n');
    end
end

fprintf('SE result:   k1=%.4f  k5=%.4f  k6=%.4f  resnorm=%.6f\n', ...
    best_p_opt_se(1), best_p_opt_se(2), best_p_opt_se(3), best_resnorm_se);

% Simulate single-expert trajectory
sim_data.glucose = G_obs;
sim_data.insulin = I_obs;
sim_data.BW      = BW;
sim_data.meal.G  = 75000;

p_vec       = EDES_Parameters(best_p_opt_se, sim_data, 1);
[x0, c_con] = EDES_Initial(sim_data, 1, p_vec);

t_saved    = 0;
G_PL_saved = G_b;

ODE_opts = odeset('RelTol', 1e-5, 'AbsTol', 1e-8, 'OutputFcn', @integratorfunG);

try
    [~, X_se] = ode45(@EDES_ODE, time_full, x0, ODE_opts, p_vec, c_con, sim_data, 1);
    if size(X_se,1) == length(time_full)
        G_se = X_se(:,2);
        I_se = X_se(:,4);
    else
        G_se = G_b    * ones(length(time_full), 1);
        I_se = I_PL_b * ones(length(time_full), 1);
    end
catch
    G_se = G_b    * ones(length(time_full), 1);
    I_se = I_PL_b * ones(length(time_full), 1);
end

%% =========================================================================
%% Performance metrics — RMSE at sparse time points
%% =========================================================================
G_moe_sp = G_moe(sparse_idx);   % [5 x 1]
I_moe_sp = I_moe(sparse_idx);
G_se_sp  = G_se(sparse_idx);
I_se_sp  = I_se(sparse_idx);

G_rmse_moe = sqrt(mean((G_moe_sp - G_obs').^2));
I_rmse_moe = sqrt(mean((I_moe_sp - I_obs').^2));
G_rmse_se  = sqrt(mean((G_se_sp  - G_obs').^2));
I_rmse_se  = sqrt(mean((I_se_sp  - I_obs').^2));

fprintf('\n--- RMSE at [0 30 60 90 120] min ---\n');
fprintf('MoE:            G_RMSE = %.4f mmol/L   I_RMSE = %.2f mU/L\n', G_rmse_moe, I_rmse_moe);
fprintf('Single-expert:  G_RMSE = %.4f mmol/L   I_RMSE = %.2f mU/L\n', G_rmse_se,  I_rmse_se);

%% =========================================================================
%% Figure 1 — Trajectory comparison
%% =========================================================================
figure('Name', 'Trajectory comparison', 'Position', [80 80 1100 460]);

subplot(1,2,1);
hold on;
plot(time_full, G_moe, '-',  'Color', col_moe, 'LineWidth', 2.4, 'DisplayName', 'MoE predicted');
plot(time_full, G_se,  '-',  'Color', col_se,  'LineWidth', 2.4, 'DisplayName', 'Single-expert predicted');
plot(t_sparse,  G_obs, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k', 'DisplayName', 'Observed');
xlabel('Time (min)'); ylabel('Glucose (mmol/L)');
title(sprintf('Glucose  —  %s patient %d (Japan)\nMoE: k1=%.4f  k5=%.4f     SE: k1=%.4f  k5=%.4f  k6=%.4f', ...
    POPULATION, PATIENT_IDX, ...
    k_opt_moe(1), k_opt_moe(2), ...
    best_p_opt_se(1), best_p_opt_se(2), best_p_opt_se(3)));
legend('Location', 'best'); grid on;

subplot(1,2,2);
hold on;
plot(time_full, I_moe, '-',  'Color', col_moe, 'LineWidth', 2.4, 'DisplayName', 'MoE predicted');
plot(time_full, I_se,  '-',  'Color', col_se,  'LineWidth', 2.4, 'DisplayName', 'Single-expert predicted');
plot(t_sparse,  I_obs, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'k', 'DisplayName', 'Observed');
xlabel('Time (min)'); ylabel('Insulin (mU/L)');
title(sprintf('Insulin  —  %s patient %d (Japan)', POPULATION, PATIENT_IDX));
legend('Location', 'best'); grid on;

sgtitle(sprintf('MoE vs Single-Expert  |  Patient %d  |  gating: w = [%.2f  %.2f  %.2f]  (NGT/IGT/T2DM)', ...
    PATIENT_IDX, w(1), w(2), w(3)));

%% =========================================================================
%% Figure 2 — Performance comparison (RMSE + parameters)
%% =========================================================================
figure('Name', 'Performance comparison', 'Position', [80 600 1000 400]);

% RMSE
subplot(1,2,1);
hold on;
rmse_vals = [G_rmse_moe, G_rmse_se; I_rmse_moe, I_rmse_se];
b1 = bar(rmse_vals, 'grouped');
b1(1).FaceColor = col_moe;
b1(2).FaceColor = col_se;
set(gca, 'XTickLabel', {'Glucose (mmol/L)', 'Insulin (mU/L)'});
ylabel('RMSE');
title('Fit accuracy (RMSE at sparse time points)');
legend({'MoE', 'Single-expert'}, 'Location', 'best');
grid on;

% Fitted parameters k1 and k5
subplot(1,2,2);
hold on;
par_vals = [k_opt_moe(1), best_p_opt_se(1); ...
            k_opt_moe(2), best_p_opt_se(2)];
b2 = bar(par_vals, 'grouped');
b2(1).FaceColor = col_moe;
b2(2).FaceColor = col_se;
set(gca, 'XTickLabel', {'k_1  (min^{-1})', 'k_5  (min^{-1})'});
ylabel('Parameter value');
%title(sprintf('Fitted parameters\n(k_6 optimised by SE only: k_6 = %.4f)', best_p_opt_se(3)));
legend({'MoE', 'Single-expert'}, 'Location', 'best');
grid on;

sgtitle(sprintf('Performance comparison  |  Patient %d  (%s)', PATIENT_IDX, POPULATION));

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
