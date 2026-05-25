%% final_architecture_dataset.m
% Runs the MoE personalised pipeline on all N=118 real patients from
% japan_population_labelled.mat.
%
%   Figure 1 — RMSE boxplots + jittered points (glucose / insulin) per category
%   Figure 2 — Mean ± 1 SD trajectories per category (observed vs MoE)
%   Figure 3 — Mean gating weights assigned per true ADA category
%
% Prerequisites: japan_population_labelled.mat, gating_weights.mat,
%                EDES_ODE.m, integratorfunG.m

clear; clc;
global t_saved G_PL_saved

%% -------------------------------------------------------------------------
%% PID expert parameters [k5, k6, k8] and fixed EDES parameters
%% -------------------------------------------------------------------------
pids = [  0.092,   0.079,   7.394;   % NGT
          0.006,   0.089,   4.724;   % IGT
          0.014,   0.000,   5.755];  % T2DM

expert_names = {'NGT', 'IGT', 'T2DM'};
colors       = {[0.18 0.63 0.18], [0.93 0.69 0.13], [0.80 0.15 0.15]};

fixed.k2      = 0.28;    fixed.k3  = 6.07e-3;  fixed.k4  = 2.35e-4;
fixed.k7      = 1.15;    fixed.k9  = 3.83e-2;  fixed.k10 = 2.84e-1;
fixed.sigma   = 1.4;     fixed.KM  = 13.2;     fixed.G_liv_b = 0.043;

%% -------------------------------------------------------------------------
%% Load dataset and gating network weights
%% -------------------------------------------------------------------------
fprintf('Loading data...\n');
raw = load('japan_population_labelled.mat');
jp  = raw.japan_population;
N   = jp.n_valid;                          % 118

gw     = load('gating_weights.mat');
W1 = gw.W1;  b1 = gw.b1(:);
W2 = gw.W2;  b2 = gw.b2(:);
W3 = gw.W3;  b3 = gw.b3(:);
X_mean = gw.X_mean(:)';
X_std  = gw.X_std(:)';

t_sparse   = double(jp.time);              % [0 30 60 90 120]
time_full  = (0:1:120)';                   % simulation grid
sparse_idx = arrayfun(@(t) find(time_full == t, 1), t_sparse);
meal_G     = 75000;                        % 75 g OGTT

%% -------------------------------------------------------------------------
%% Preallocate result arrays
%% -------------------------------------------------------------------------
k1_all    = zeros(N,1);
k5_all    = zeros(N,1);
w_all     = zeros(N,3);
cats      = zeros(N,1);   % 1=NGT  2=IGT  3=T2DM
BW_all    = zeros(N,1);
G_obs_all = zeros(N,5);
I_obs_all = zeros(N,5);
G_pred_sp = zeros(N,5);   % MoE prediction at sparse time points
I_pred_sp = zeros(N,5);

opt_options = optimoptions('lsqnonlin', 'Display', 'off', ...
    'MaxFunctionEvaluations', 1000, 'FunctionTolerance', 1e-8);

%% -------------------------------------------------------------------------
%% Main optimisation loop
%% -------------------------------------------------------------------------
fprintf('Optimising %d patients...\n', N);
for i = 1:N
    G_obs  = jp.glucose_noisy(i, :);
    I_obs  = jp.insulin_noisy(i, :);
    BW     = jp.BW(i);
    G_b    = G_obs(1);
    I_PL_b = I_obs(1);

    G_obs_all(i,:) = G_obs;
    I_obs_all(i,:) = I_obs;
    BW_all(i)      = BW;

    if      jp.is_NGT(i),  cats(i) = 1;
    elseif  jp.is_IGT(i),  cats(i) = 2;
    elseif  jp.is_T2DM(i), cats(i) = 3;
    end

    % Gating network forward pass
    x_norm = ([G_obs, I_obs] - X_mean) ./ X_std;
    h1 = max(0, W1 * x_norm' + b1);
    h2 = max(0, W2 * h1      + b2);
    z  = W3 * h2 + b3;
    e_z = exp(z - max(z));
    w   = double(e_z / sum(e_z));   % [3 x 1]
    w_all(i,:) = w';

    % Optimise [k1, k5]
    k0 = double([0.028, w' * pids(:,1)]);
    % lb = [0.0,  0.0 ];
    % ub = [inf,  inf];
    lb = [0.0, 0.0];
    ub = [0.05, 0.17];

    try
        [k_opt, ~] = lsqnonlin( ...
            @(k) weighted_residuals(k, w, pids, fixed, G_b, I_PL_b, BW, meal_G, ...
                                    time_full, sparse_idx, G_obs, I_obs), ...
            k0, lb, ub, opt_options);
    catch
        k_opt = k0;
        fprintf('  Patient %d: optimisation failed — using initial guess\n', i);
    end
    k1_all(i) = k_opt(1);
    k5_all(i) = k_opt(2);

    % MoE prediction at sparse time points (for RMSE)
    G_p = zeros(1, numel(sparse_idx));
    I_p = zeros(1, numel(sparse_idx));
    for e = 1:3
        [G_sim, I_sim] = run_simulation(k_opt(1), k_opt(2), pids(e,2), pids(e,3), ...
            time_full, fixed, G_b, I_PL_b, BW, meal_G);
        G_p = G_p + w(e) * G_sim(sparse_idx)';
        I_p = I_p + w(e) * I_sim(sparse_idx)';
    end
    G_pred_sp(i,:) = G_p;
    I_pred_sp(i,:) = I_p;

    if mod(i,20) == 0
        fprintf('  %d / %d\n', i, N);
    end
end
fprintf('Optimisation complete.\n\n');

save('dataset_results.mat', 'k1_all', 'k5_all', 'cats', 'w_all');
fprintf('Results saved to dataset_results.mat\n\n');

%% -------------------------------------------------------------------------
%% Full MoE trajectories on time_full (for smooth mean-curve plots)
%% -------------------------------------------------------------------------
G_traj = zeros(N, length(time_full));
I_traj = zeros(N, length(time_full));

fprintf('Simulating full trajectories...\n');
for i = 1:N
    w_i    = w_all(i,:)';
    G_b    = G_obs_all(i,1);
    I_PL_b = I_obs_all(i,1);
    BW     = BW_all(i);
    G_p    = zeros(length(time_full), 1);
    I_p    = zeros(length(time_full), 1);
    for e = 1:3
        [G_sim, I_sim] = run_simulation(k1_all(i), k5_all(i), pids(e,2), pids(e,3), ...
            time_full, fixed, G_b, I_PL_b, BW, meal_G);
        G_p = G_p + w_i(e) * G_sim;
        I_p = I_p + w_i(e) * I_sim;
    end
    G_traj(i,:) = G_p';
    I_traj(i,:) = I_p';
end
fprintf('Done.\n\n');

%% -------------------------------------------------------------------------
%% Per-category indices and RMSE
%% -------------------------------------------------------------------------
idx_cat = {find(cats==1), find(cats==2), find(cats==3)};

G_rmse = sqrt(mean((G_pred_sp - G_obs_all).^2, 2));
I_rmse = sqrt(mean((I_pred_sp - I_obs_all).^2, 2));

fprintf('--- RMSE Summary ---\n');
for c = 1:3
    idx = idx_cat{c};
    fprintf('%s (n=%d):  G_RMSE = %.3f +/- %.3f mmol/L   I_RMSE = %.2f +/- %.2f mU/L\n', ...
        expert_names{c}, numel(idx), ...
        mean(G_rmse(idx)), std(G_rmse(idx)), ...
        mean(I_rmse(idx)), std(I_rmse(idx)));
end
fprintf('\n');

%% =========================================================================
%% Figure 1 — RMSE distributions per ADA category
%% =========================================================================
data_G = vertcat(G_rmse(idx_cat{1}), G_rmse(idx_cat{2}), G_rmse(idx_cat{3}));
data_I = vertcat(I_rmse(idx_cat{1}), I_rmse(idx_cat{2}), I_rmse(idx_cat{3}));
grp    = vertcat( ones(numel(idx_cat{1}),1), ...
                 2*ones(numel(idx_cat{2}),1), ...
                 3*ones(numel(idx_cat{3}),1) );

figure('Name', 'Fit Accuracy', 'Position', [50 50 900 420]);

subplot(1,2,1);
hold on;
boxplot(data_G, grp, 'Labels', expert_names, 'Widths', 0.45, 'Symbol', '');
for c = 1:3
    idx = idx_cat{c};
    jx  = c + 0.18*(rand(numel(idx),1)-0.5);
    scatter(jx, G_rmse(idx), 28, 'filled', 'MarkerFaceColor', colors{c}, ...
        'MarkerFaceAlpha', 0.55, 'HandleVisibility', 'off');
end
ylabel('RMSE (mmol/L)'); title('Glucose fit RMSE'); grid on;

subplot(1,2,2);
hold on;
boxplot(data_I, grp, 'Labels', expert_names, 'Widths', 0.45, 'Symbol', '');
for c = 1:3
    idx = idx_cat{c};
    jx  = c + 0.18*(rand(numel(idx),1)-0.5);
    scatter(jx, I_rmse(idx), 28, 'filled', 'MarkerFaceColor', colors{c}, ...
        'MarkerFaceAlpha', 0.55, 'HandleVisibility', 'off');
end
ylabel('RMSE (mU/L)'); title('Insulin fit RMSE'); grid on;

sgtitle('MoE Fit Accuracy by ADA Category  —  Japan Dataset');

%% =========================================================================
%% Figure 2 — Mean ± 1 SD trajectories per ADA category
%% =========================================================================
figure('Name', 'Mean trajectories', 'Position', [50 530 1300 700]);

for c = 1:3
    idx = idx_cat{c};
    clr = colors{c};

    G_obs_mu  = mean(G_obs_all(idx,:), 1);
    G_obs_sd  = std(G_obs_all(idx,:),  0, 1);
    I_obs_mu  = mean(I_obs_all(idx,:), 1);
    I_obs_sd  = std(I_obs_all(idx,:),  0, 1);
    G_traj_mu = mean(G_traj(idx,:), 1);
    I_traj_mu = mean(I_traj(idx,:), 1);

    % Glucose
    subplot(2,3,c);
    hold on;
    fill([t_sparse, fliplr(t_sparse)], ...
         [G_obs_mu+G_obs_sd, fliplr(G_obs_mu-G_obs_sd)], ...
         clr, 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(t_sparse,  G_obs_mu,  'o-', 'Color', clr, 'LineWidth', 2.0, ...
         'MarkerFaceColor', clr, 'MarkerSize', 6, 'DisplayName', 'Observed (mean +/-1SD)');
    plot(time_full, G_traj_mu, 'k-', 'LineWidth', 2.0, 'DisplayName', 'MoE predicted');
    xlabel('Time (min)'); ylabel('Glucose (mmol/L)');
    title(sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
    legend('Location', 'best'); grid on;

    % Insulin
    subplot(2,3,c+3);
    hold on;
    fill([t_sparse, fliplr(t_sparse)], ...
         [I_obs_mu+I_obs_sd, fliplr(I_obs_mu-I_obs_sd)], ...
         clr, 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(t_sparse,  I_obs_mu,  'o-', 'Color', clr, 'LineWidth', 2.0, ...
         'MarkerFaceColor', clr, 'MarkerSize', 6, 'DisplayName', 'Observed (mean +/-1SD)');
    plot(time_full, I_traj_mu, 'k-', 'LineWidth', 2.0, 'DisplayName', 'MoE predicted');
    xlabel('Time (min)'); ylabel('Insulin (mU/L)');
    title(sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
    legend('Location', 'best'); grid on;
end
sgtitle('Mean +/- 1 SD: Observed vs MoE Predicted  —  Japan Dataset');

%% =========================================================================
%% Figure 3 — Gating weight distribution per ADA category
%% =========================================================================
figure('Name', 'Gating weights', 'Position', [1000 50 720 460]);
hold on;
x = 1:3;  bar_w = 0.25;
expert_labels = {'w_{NGT}', 'w_{IGT}', 'w_{T2DM}'};

for e = 1:3
    means = zeros(1,3);
    errs  = zeros(1,3);
    for c = 1:3
        idx = idx_cat{c};
        means(c) = mean(w_all(idx,e));
        errs(c)  = std(w_all(idx,e));
    end
    bar(x + (e-2)*bar_w, means, bar_w, 'FaceColor', colors{e}, ...
        'FaceAlpha', 0.85, 'DisplayName', expert_labels{e});
    errorbar(x + (e-2)*bar_w, means, errs, 'k.', 'LineWidth', 1.4, ...
        'HandleVisibility', 'off');
end
xticks(x);  xticklabels(expert_names);
ylabel('Mean gating weight');  ylim([0 1]);
title('Mean gating weights per true ADA category  —  Japan Dataset');
legend('Location', 'best');  grid on;


%% =========================================================================
%% Local functions (identical to final_architecture_real.m)
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
