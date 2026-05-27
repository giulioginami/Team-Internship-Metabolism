%% PID_optimization_full.m
% Optimizes k5, k6, k8 using the full EDES ODE model.
% Fits to the median glucose and insulin trajectories of each population.

clear; clc;
global t_saved G_PL_saved

%% Load data
data = load('virtual_population_labelled.mat');
populations = {'NGT', 'IGT', 'T2DM'};
time   = (0:1:480)';
meal_G = 75000;

%% Fixed parameters (Rozendaal et al. 2018)
fixed.k2      = 0.28;
fixed.k3      = 6.07e-3;
fixed.k4      = 2.35e-4;
fixed.k7      = 1.15;
fixed.k9      = 3.83e-2;
fixed.k10     = 2.84e-1;
fixed.sigma   = 1.4;
fixed.KM      = 13.2;
fixed.G_liv_b = 0.043;

%% Population quality check
fprintf('=== Population sizes ===\n');
for p = 1:3
    pop = populations{p};
    fprintf('  %s : %d patients\n', pop, data.(['dataset_' pop]).n);
end
fprintf('\n');

results = struct();

for p = 1:3
    pop     = populations{p};
    dataset = data.(['dataset_' pop]);

    % Median trajectories (noisy data)
    G_med = median(dataset.glucose_noisy, 1)';
    I_med = median(dataset.insulin_noisy, 1)';

    % Representative patient: median basal values and body parameters
    G_b    = G_med(1);
    I_PL_b = I_med(1);
    BW     = median(dataset.param_matrix(:, 7));
    k1     = median(dataset.param_matrix(:, 1));

    % Initial guess: median of individual patient parameters
    k_init = [median(dataset.param_matrix(:, 2)), ...  % k5
              median(dataset.param_matrix(:, 3)), ...  % k6
              median(dataset.param_matrix(:, 4))];     % k8

    lb = [0,    0,    0   ];
    ub = [0.17, 0.34, 10.0];

    options = optimoptions('lsqnonlin', 'Display', 'iter', 'MaxFunctionEvaluations', 500);

    fprintf('\n=== Optimizing %s ===\n', pop);

    [k_opt, resnorm] = lsqnonlin( ...
        @(k) compute_residuals(k, G_med, I_med, time, k1, fixed, G_b, I_PL_b, BW, meal_G), ...
        k_init, lb, ub, options);

    fprintf('k5 = %.4f  |  k6 = %.4f  |  k8 = %.4f  |  resnorm = %.6f\n', ...
        k_opt(1), k_opt(2), k_opt(3), resnorm);

    % Final simulation with optimal parameters
    [G_sim, I_sim] = run_simulation(k_opt, time, k1, fixed, G_b, I_PL_b, BW, meal_G);

    results.(pop).k_opt = k_opt;
    results.(pop).G_sim = G_sim;
    results.(pop).I_sim = I_sim;
    results.(pop).G_med = G_med;
    results.(pop).I_med = I_med;
end

%% Plot results
colors = struct('NGT',  [0.18 0.63 0.18], ...
                'IGT',  [0.93 0.69 0.13], ...
                'T2DM', [0.80 0.15 0.15]);

% Compute shared y-axis limits across all populations
all_G = cellfun(@(p) [results.(p).G_med; results.(p).G_sim], populations, 'UniformOutput', false);
all_I = cellfun(@(p) [results.(p).I_med; results.(p).I_sim], populations, 'UniformOutput', false);
ylim_G = [0, max(cellfun(@max, all_G)) * 1.1];
ylim_I = [0, max(cellfun(@max, all_I)) * 1.1];

figure('Name', 'PID Optimization - Full EDES Model', 'Position', [100 100 1200 500]);

for p = 1:3
    pop = populations{p};
    clr = colors.(pop);
    r   = results.(pop);

    subplot(2, 3, p);
    plot(time, r.G_med, '--', 'Color', clr,      'LineWidth', 1.5); hold on;
    plot(time, r.G_sim, '-',  'Color', clr * 0.6, 'LineWidth', 2);
    xlabel('Time (min)'); ylabel('Glucose (mmol/L)');
    title(sprintf('%s  |  k5=%.3f  k6=%.3f  k8=%.3f', pop, r.k_opt(1), r.k_opt(2), r.k_opt(3)));
    legend('Median data', 'Optimized', 'Location', 'best');
    ylim(ylim_G); grid on;

    subplot(2, 3, p + 3);
    plot(time, r.I_med, '--', 'Color', clr,      'LineWidth', 1.5); hold on;
    plot(time, r.I_sim, '-',  'Color', clr * 0.6, 'LineWidth', 2);
    xlabel('Time (min)'); ylabel('Insulin (mU/L)');
    title(sprintf('%s - Insulin', pop));
    legend('Median data', 'Optimized', 'Location', 'best');
    ylim(ylim_I); grid on;
end

sgtitle('PID Optimization - Full EDES Model');

%% -------------------------------------------------------------------------
%% Helper functions
%% -------------------------------------------------------------------------

function res = compute_residuals(k, G_med, I_med, time, k1, fixed, G_b, I_PL_b, BW, meal_G)
    [G_sim, I_sim] = run_simulation(k, time, k1, fixed, G_b, I_PL_b, BW, meal_G);
    res = [(G_sim - G_med) / norm(G_med);
           (I_sim - I_med) / norm(I_med)];
end

function [G_sim, I_sim] = run_simulation(k, time, k1, fixed, G_b, I_PL_b, BW, meal_G)
    global t_saved G_PL_saved

    k5 = k(1);
    k6 = k(2);
    k8 = k(3);

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
