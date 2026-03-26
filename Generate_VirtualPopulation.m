%% Generate_VirtualPopulation.m
% Generates in silico postprandial glucose and insulin trajectories for a
% large heterogeneous virtual population using the EDES model.
%
% Key varied parameters (most sensitive, from sensitivity analysis):
%   k1  - rate constant of glucose appearance in the gut
%   k5  - delayed insulin-dependent glucose uptake
%   k6  - proportional beta-cell response (P-gain)
%   k8  - derivative beta-cell response (D-gain)
%   G_b - fasting plasma glucose
%   I_PL_b - fasting plasma insulin
%   BW  - body weight
%
% All other parameters fixed to Rozendaal et al. (2018) values.
% Meal: 75g OGTT (standard oral glucose tolerance test)
% Time: 0:1:480 minutes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;
rng(42); % reproducibility

%% ========================================================================
% Settings
% =========================================================================
N        = 5000;           % number of virtual individuals to attempt
time     = (0:1:480)';     % simulation time vector [min]
meal_G   = 75000;          % 75g OGTT in mg

%% ========================================================================
% Parameter ranges for Latin Hypercube Sampling
% Columns: [lower_bound, upper_bound]
% Order:   k1, k5, k6, k8, G_b, I_PL_b, BW
% =========================================================================
param_bounds = [
    0.02,    0.03;    % k1  [1/min]
    0.0,    0.17;    % k5  [1/min]
    0.0,    0.34;     % k6  [-]
    0.0,    10.0;    % k8  [-]
    3.9,    12.0;     % G_b [mmol/L]
    2.0,    55.6;    % I_PL_b [mU/L]
    60.0,   130.0;   % BW  [kg]
];

n_params = size(param_bounds, 1); % = 7

%% ========================================================================
% Latin Hypercube Sampling  (requires Statistics and Machine Learning Toolbox)
% =========================================================================
lhs_unit = lhsdesign(N, n_params); % N x 7 matrix in [0,1]

% Scale to physical ranges -> value = normalized * (max - min) + min
lhs_scaled = lhs_unit .* (param_bounds(:,2) - param_bounds(:,1))' + param_bounds(:,1)';

% Unpack columns
k1_vec     = lhs_scaled(:,1);
k5_vec     = lhs_scaled(:,2);
k6_vec     = lhs_scaled(:,3);
k8_vec     = lhs_scaled(:,4);
Gb_vec     = lhs_scaled(:,5);
Ib_vec     = lhs_scaled(:,6);
BW_vec     = lhs_scaled(:,7);

%% ========================================================================
% Fixed parameters (Rozendaal et al. 2018)
% =========================================================================
k2     = 0.28;       % rate constant of gut emptying [1/min]
k3     = 6.07e-3;    % hepatic glucose release suppression by G [1/min]
k4     = 2.35e-4;    % hepatic glucose release suppression by delayed I [1/min]
k7     = 1.15;       % integral beta-cell gain [1/min]
k9     = 3.83e-2;    % insulin plasma->remote rate [1/min]
k10    = 2.84e-1;    % insulin remote degradation [1/min]
sigma  = 1.4;        % meal appearance shape factor [-]
KM     = 13.2;       % Michaelis-Menten coefficient for glucose uptake [mmol/L]
G_liv_b_fixed = 0.043; % basal hepatic glucose release

%% ========================================================================
% Pre-allocate output arrays
% =========================================================================
n_t = length(time);

glucose_clean  = NaN(N, n_t);  % noise-free glucose [mmol/L]
insulin_clean  = NaN(N, n_t);  % noise-free insulin [mU/L]
glucose_noisy  = NaN(N, n_t);  % glucose + measurement noise
insulin_noisy  = NaN(N, n_t);  % insulin + measurement noise
param_matrix   = NaN(N, 7);    % sampled parameters for each accepted individual
valid_flag     = false(N, 1);  % QC pass/fail

%% ========================================================================
% Main simulation loop
% =========================================================================
fprintf('Starting virtual population generation (N=%d)...\n', N);

global t_saved G_PL_saved  % required by EDES_ODE via integratorfunG

for i = 1:N

    if mod(i,100) == 0
        n_valid = sum(valid_flag);
        fprintf('  Individual %d/%d | Valid so far: %d\n', i, N, n_valid);
    end

    %% Build full 15-element parameter vector
    k1     = k1_vec(i);
    k5     = k5_vec(i);
    k6     = k6_vec(i);
    k8     = k8_vec(i);
    G_b    = Gb_vec(i);
    I_PL_b = Ib_vec(i);
    BW     = BW_vec(i);

    parameters      = zeros(1,15);
    parameters(1)   = k1;
    parameters(2)   = k2;
    parameters(3)   = k3;
    parameters(4)   = k4;
    parameters(5)   = k5;
    parameters(6)   = k6;
    parameters(7)   = k7;
    parameters(8)   = k8;
    parameters(9)   = k9;
    parameters(10)  = k10;
    parameters(11)  = sigma;
    parameters(12)  = KM;
    parameters(13)  = G_b;
    parameters(14)  = I_PL_b;
    parameters(15)  = G_liv_b_fixed;

    %% Build input_data struct (row 1)
    input_data.glucose    = G_b;
    input_data.insulin    = I_PL_b;
    input_data.BW         = BW;
    input_data.meal.G     = meal_G;

    %% Compute initial conditions and constants (inline, mirrors EDES_Initial)
    x0 = [0, G_b, 0, I_PL_b, 0]; % [M_G_gut, G_PL, G_int, I_PL, I_d1]

    c.f_G              = 0.005551;
    c.f_I              = 1;
    c.V_G              = 17/70;
    c.G_liv_b          = G_liv_b_fixed;
    c.tau_i            = 31;
    c.tau_d            = 3;
    c.G_th_PL          = 9;
    c.t_integralwindow = 30;
    c.c1               = 0.1;
    c.c2 = G_liv_b_fixed .* (KM + G_b) ./ G_b - k5 .* c.f_I .* G_liv_b_fixed;
    c.c3 = k7 .* G_b ./ (c.f_I * c.tau_i .* I_PL_b) .* c.t_integralwindow;

    %% Initialise global integrator state (required by EDES_ODE)
    t_saved     = 0;
    G_PL_saved  = G_b;

    %% Run ODE solver
    ODE_options = odeset('RelTol',1e-5, 'AbsTol',1e-8, ...
                         'OutputFcn',@integratorfunG);
    try
        [T, X] = ode45(@EDES_ODE, time, x0, ODE_options, ...
                        parameters, c, input_data, 1);
    catch
        % ODE solver failure -> discard
        continue
    end

    % Verify solver returned values at all requested time points
    if length(T) ~= n_t
        continue
    end

    G_sim = X(:,2); % plasma glucose [mmol/L]
    I_sim = X(:,4); % plasma insulin [mU/L]

    %% Quality control filters
    % 1) No negative values
    if any(G_sim < 0) || any(I_sim < 0)
        continue
    end

    % 2) Glucose must stay within physiologically plausible range
    if max(G_sim) > 30 || min(G_sim) < 2
        continue
    end

    % 3) Insulin must stay within physiologically plausible range
    if max(I_sim) > 400 || min(I_sim) < 0
        continue
    end

%     % 4) Peak glucose must occur within first 240 min (postprandial window)
%     [~, peak_idx] = max(G_sim);
%     if peak_idx > 241   % index 241 = 240 min (0-based time)
%         continue
%     end
% 
%     % 5) Glucose must return towards baseline (no runaway trajectories)
%     if G_sim(end) > G_b + 5
%         continue
%     end

    %% Add realistic measurement noise
    noise_G_pct = 0.02 + 0.01  * rand();   % uniform 2-3%
    noise_I_pct = 0.05 + 0.03  * rand();   % uniform 5-8%

    G_noisy = G_sim .* (1 + noise_G_pct * randn(n_t, 1));
    I_noisy = I_sim .* (1 + noise_I_pct * randn(n_t, 1));

    % Clamp noisy values to be non-negative
    G_noisy = max(G_noisy, 0);
    I_noisy = max(I_noisy, 0);

    %% Store accepted individual
    glucose_clean(i,:)  = G_sim';
    insulin_clean(i,:)  = I_sim';
    glucose_noisy(i,:)  = G_noisy';
    insulin_noisy(i,:)  = I_noisy';
    param_matrix(i,:)   = [k1, k5, k6, k8, G_b, I_PL_b, BW];
    valid_flag(i)        = true;

end

%% ========================================================================
% Trim to accepted individuals only
% =========================================================================
idx_valid = find(valid_flag);
n_valid   = length(idx_valid);
fprintf('\nSimulation complete. Accepted: %d / %d individuals (%.1f%%)\n', ...
        n_valid, N, 100*n_valid/N);

glucose_clean  = glucose_clean(idx_valid, :);
insulin_clean  = insulin_clean(idx_valid, :);
glucose_noisy  = glucose_noisy(idx_valid, :);
insulin_noisy  = insulin_noisy(idx_valid, :);
param_matrix   = param_matrix(idx_valid, :);

%% ========================================================================
% Save results
% =========================================================================
virtual_population.time           = time';
virtual_population.glucose_clean  = glucose_clean;   % [n_valid x n_t]
virtual_population.insulin_clean  = insulin_clean;   % [n_valid x n_t]
virtual_population.glucose_noisy  = glucose_noisy;   % [n_valid x n_t]
virtual_population.insulin_noisy  = insulin_noisy;   % [n_valid x n_t]
virtual_population.param_matrix   = param_matrix;    % [n_valid x 7]
virtual_population.param_names    = {'k1','k5','k6','k8','G_b','I_PL_b','BW'};
virtual_population.n_valid        = n_valid;
virtual_population.N_attempted    = N;

save('virtual_population.mat', 'virtual_population', '-v7.3');
fprintf('Saved: virtual_population.mat\n');

%% ========================================================================
% Quick summary plot
% =========================================================================
t_plot = time;
n_plot = min(200, n_valid);
idx_plot = randperm(n_valid, n_plot);

figure('Name','Virtual Population - Overview','Position',[100 100 1200 500]);

subplot(1,2,1);
plot(t_plot, glucose_noisy(idx_plot,:)', 'Color', [0.2 0.5 0.8 0.15], 'LineWidth',0.5);
hold on;
plot(t_plot, median(glucose_noisy,1), 'k-', 'LineWidth', 2);
xlabel('Time (min)'); ylabel('Glucose (mmol/L)');
title(sprintf('Plasma Glucose  (n=%d shown)', n_plot));
xlim([0 480]); grid on;

subplot(1,2,2);
plot(t_plot, insulin_noisy(idx_plot,:)', 'Color', [0.8 0.3 0.2 0.15], 'LineWidth',0.5);
hold on;
plot(t_plot, median(insulin_noisy,1), 'k-', 'LineWidth', 2);
xlabel('Time (min)'); ylabel('Insulin (mU/L)');
title(sprintf('Plasma Insulin  (n=%d shown)', n_plot));
xlim([0 480]); grid on;

sgtitle(sprintf('EDES Virtual Population  |  N_{valid} = %d / %d', n_valid, N));
