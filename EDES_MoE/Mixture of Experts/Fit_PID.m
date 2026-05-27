%% 
% Fits the single-expert EDES model to sparse OGTT data from one real patient
% in japan_population_labelled.mat
%
%   1. Optimise [k1, k5, k6] via LHS multi-start lsqnonlin on EDES_ErrorFunc
%   2. Plot: fitted EDES curve (solid) + observed data
%
% Optimisation settings:
%   - 3 free parameters:  [k1, k5, k6]
%   - k8 fixed at 7.27 (via EDES_Parameters.m)
%   - bounds:  lb=[0.005, 0, 0]   ub=[0.1, 1, 15]
%   - num_par_sets LHS starts, best selected by minimum resnorm
%   - EDES_ErrorFunc: error normalised by max(obs) + AUC regularisation
%   - Optimisation time: 0:1:240  (AUC term requires X(1:240,1))
%
% Functions used:
%   EDES_ErrorFunc.m, EDES_Parameters.m, EDES_Initial.m,
%   EDES_ODE.m, integratorfunG.m
%
% Prerequisites: japan_population_labelled.mat

clear; clc;
global t_saved G_PL_saved

%% -------------------------------------------------------------------------
%% Settings
%% -------------------------------------------------------------------------
PATIENT_IDX  = 4;   % 1 to 118
num_par_sets = 5;     % number of LHS starting points

%% -------------------------------------------------------------------------
%% Load dataset and extract patient
%% -------------------------------------------------------------------------
fprintf('Loading japan_population_labelled.mat...\n');
raw = load('japan_population_labelled.mat');
jp  = raw.japan_population;

t_sparse = double(jp.time);   % [0 30 60 90 120]

G_obs  = jp.glucose_noisy(PATIENT_IDX, :);   % [1 x 5]  mmol/L
I_obs  = jp.insulin_noisy(PATIENT_IDX, :);   % [1 x 5]  mU/L
BW     = jp.BW(PATIENT_IDX);

if      jp.is_NGT(PATIENT_IDX),  POPULATION = 'NGT';
elseif  jp.is_IGT(PATIENT_IDX),  POPULATION = 'IGT';
elseif  jp.is_T2DM(PATIENT_IDX), POPULATION = 'T2DM';
else,                             POPULATION = 'unclassified';
end

fprintf('Patient %d  |  ADA category: %s  |  G_b=%.2f  I_PL_b=%.2f  BW=%.1f\n', ...
    PATIENT_IDX, POPULATION, G_obs(1), I_obs(1), BW);

%% -------------------------------------------------------------------------
%% Build input_data struct (compatible with EDES_ErrorFunc)
%% -------------------------------------------------------------------------
input_data.glucose = G_obs;       % [1 x 5]  mmol/L
input_data.insulin = I_obs;       % [1 x 5]  mU/L
input_data.BW      = BW;
input_data.meal.G  = 75000;       % 75 g OGTT in mg
input_data.time_G  = t_sparse;    % [0 30 60 90 120]
input_data.time_I  = t_sparse;    % [0 30 60 90 120]

%% -------------------------------------------------------------------------
%% Optimisation settings (from Fit_EDES_LatinHyperCube.m)
%% -------------------------------------------------------------------------
num_par = 3;              % optimised: [k1, k5, k6]
lb = [0.005, 0,  0 ];    % lower bounds (same as Fit_EDES_LatinHyperCube.m)
ub = [0.1,   1,  15];    % upper bounds
d  = ub - lb;

time_opt  = 0:1:240;      % optimisation time (AUC term uses X(1:240,1))
time_full = (0:1:240)';   % display simulation window (matches final_architecture_real.m)

lsq_options = optimset('Algorithm', 'trust-region-reflective', ...
    'MaxFunEvals', 1000, 'TolX', 1e-8, 'Display', 'off');

%% -------------------------------------------------------------------------
%% Latin Hypercube multi-start optimisation of [k1, k5, k6]
%% -------------------------------------------------------------------------
fprintf('\nOptimising [k1, k5, k6] with %d LHS starts...\n', num_par_sets);

lhs_samples  = lhsdesign(num_par_sets, num_par);
initial_pars = lhs_samples .* d + lb;

best_resnorm = inf;
best_p_opt   = initial_pars(1,:);

for s = 1:num_par_sets
    fprintf('  Start %d/%d  (k1=%.4f  k5=%.4f  k6=%.4f) ...', ...
        s, num_par_sets, initial_pars(s,1), initial_pars(s,2), initial_pars(s,3));
    try
        [p_opt_s, resnorm_s] = lsqnonlin(@EDES_ErrorFunc, initial_pars(s,:), ...
            lb, ub, lsq_options, input_data, 1, time_opt);
        fprintf('  resnorm = %.6f\n', resnorm_s);
        if resnorm_s < best_resnorm
            best_resnorm = resnorm_s;
            best_p_opt   = p_opt_s;
        end
    catch
        fprintf('  FAILED\n');
    end
end

fprintf('\nBest:  k1=%.4f  k5=%.4f  k6=%.4f  resnorm=%.6f\n', ...
    best_p_opt(1), best_p_opt(2), best_p_opt(3), best_resnorm);

%% -------------------------------------------------------------------------
%% Simulate with best parameters on time_full for plotting
%% -------------------------------------------------------------------------
sim_data.glucose = G_obs;
sim_data.insulin = I_obs;
sim_data.BW      = BW;
sim_data.meal.G  = 75000;

p_vec       = EDES_Parameters(best_p_opt, sim_data, 1);
[x0, c_con] = EDES_Initial(sim_data, 1, p_vec);

t_saved    = 0;
G_PL_saved = G_obs(1);

ODE_opts = odeset('RelTol', 1e-5, 'AbsTol', 1e-8, 'OutputFcn', @integratorfunG);

[~, X] = ode45(@EDES_ODE, time_full, x0, ODE_opts, p_vec, c_con, sim_data, 1);
G_fit = X(:, 2);
I_fit = X(:, 4);

%% -------------------------------------------------------------------------
%% Plot
%% -------------------------------------------------------------------------
figure('Name', 'Single-Expert Personalised Parameter Estimation', ...
    'Position', [80 80 1100 460]);

subplot(1, 2, 1);
hold on;
plot(time_full, G_fit, 'k-',  'LineWidth', 2.4, 'DisplayName', 'Single-expert fit');
plot(t_sparse,  G_obs, 'ko',  'MarkerSize', 7, 'MarkerFaceColor', 'k', ...
     'DisplayName', 'Observed');
xlabel('Time (min)'); ylabel('Glucose (mmol/L)');
title(sprintf('Glucose -- %s patient %d (Japan)\nk1=%.4f  k5=%.4f  k6=%.4f', ...
    POPULATION, PATIENT_IDX, best_p_opt(1), best_p_opt(2), best_p_opt(3)));
legend('Location', 'best'); grid on;

subplot(1, 2, 2);
hold on;
plot(time_full, I_fit, 'k-',  'LineWidth', 2.4, 'DisplayName', 'Single-expert fit');
plot(t_sparse,  I_obs, 'ko',  'MarkerSize', 7, 'MarkerFaceColor', 'k', ...
     'DisplayName', 'Observed');
xlabel('Time (min)'); ylabel('Insulin (mU/L)');
title(sprintf('Insulin -- %s patient %d (Japan)', POPULATION, PATIENT_IDX));
legend('Location', 'best'); grid on;

sgtitle(sprintf('Single-expert personalised fit  |  Patient %d  |  resnorm = %.4f', ...
    PATIENT_IDX, best_resnorm));
