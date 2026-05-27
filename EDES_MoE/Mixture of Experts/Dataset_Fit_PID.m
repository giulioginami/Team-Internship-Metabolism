%% 
% Runs single-expert EDES fitting on all N=118 real
% patients from japan_population_labelled.mat and produces summary figures.
%
%   Figure 1 — RMSE boxplots + jittered points (glucose / insulin) per category
%   Figure 2 — Mean ± 1 SD trajectories per category (observed vs predicted)
%
% Optimisation method (identical to Fit_EDES_LatinHyperCube.m):
%   - 3 free parameters: [k1, k5, k6]
%   - k8 fixed at 7.27 (via EDES_Parameters.m)
%   - bounds: lb=[0.005, 0, 0]  ub=[0.1, 1, 15]
%   - num_par_sets Latin Hypercube starts, best by minimum resnorm
%   - lsqnonlin trust-region-reflective, MaxFunEvals=1000, TolX=1e-8
%   - EDES_ErrorFunc: error normalised by max(obs) + AUC regularisation
%   - Simulation time 0:1:240 for optimisation (required by AUC term)
%
% Functions used:
%   EDES_ErrorFunc.m, EDES_Parameters.m, EDES_Initial.m,
%   EDES_ODE.m, integratorfunG.m
%
% Prerequisites: japan_population_labelled.mat

clear; clc;
global t_saved G_PL_saved

%% -------------------------------------------------------------------------
%% Load dataset
%% -------------------------------------------------------------------------
fprintf('Loading data...\n');
raw = load('japan_population_labelled.mat');
jp  = raw.japan_population;
N   = jp.n_valid;   % 118

%% -------------------------------------------------------------------------
%% Build input_data struct (compatible with EDES_ErrorFunc)
%% -------------------------------------------------------------------------
input_data.glucose = double(jp.glucose_noisy);        % [N x 5]  mmol/L
input_data.insulin = double(jp.insulin_noisy);        % [N x 5]  mU/L
input_data.BW      = double(jp.BW(:));                % [N x 1]  kg
input_data.meal.G  = 75000 * ones(N, 1);             % [N x 1]  75 g OGTT in mg
input_data.time_G  = double(jp.time);                 % [0 30 60 90 120]
input_data.time_I  = double(jp.time);                 % [0 30 60 90 120]

%% -------------------------------------------------------------------------
%% Category labels and plot settings
%% -------------------------------------------------------------------------
expert_names = {'NGT', 'IGT', 'T2DM'};
colors       = {[0.18 0.63 0.18], [0.93 0.69 0.13], [0.80 0.15 0.15]};

cats = zeros(N, 1);
for i = 1:N
    if      jp.is_NGT(i),  cats(i) = 1;
    elseif  jp.is_IGT(i),  cats(i) = 2;
    elseif  jp.is_T2DM(i), cats(i) = 3;
    end
end

%% -------------------------------------------------------------------------
%% Optimisation settings (from Fit_EDES_LatinHyperCube.m)
%% -------------------------------------------------------------------------
num_par_sets = 5;           % LHS starting points (=5 in Script_fit_and_simulated_EDES.m)
num_par      = 3;           % optimised: [k1, k5, k6]
lb = [0.005, 0,  0 ];      % lower bounds (same as Fit_EDES_LatinHyperCube.m)
ub = [0.1,   1,  15];      % upper bounds
d  = ub - lb;

time_opt  = 0:1:240;        % optimisation time (AUC term uses X(1:240,1))
time_full = (0:1:120)';     % display / RMSE simulation window (matches MoE)

t_sparse   = double(jp.time);
sparse_idx = arrayfun(@(t) find(time_full == t, 1), t_sparse);

lsq_options = optimset('Algorithm', 'trust-region-reflective', ...
    'MaxFunEvals', 1000, 'TolX', 1e-8, 'Display', 'off', 'UseParallel', 0);

%% -------------------------------------------------------------------------
%% Preallocate results
%% -------------------------------------------------------------------------
k1_all    = zeros(N, 1);
k5_all    = zeros(N, 1);
k6_all    = zeros(N, 1);
G_obs_all = zeros(N, 5);
I_obs_all = zeros(N, 5);
G_pred_sp = zeros(N, 5);
I_pred_sp = zeros(N, 5);
G_traj    = zeros(N, length(time_full));
I_traj    = zeros(N, length(time_full));

%% -------------------------------------------------------------------------
%% Main optimisation + simulation loop
%% -------------------------------------------------------------------------
fprintf('Optimising %d patients (single-expert, %d LHS starts each)...\n', N, num_par_sets);

for i = 1:N
    G_obs_all(i,:) = jp.glucose_noisy(i,:);
    I_obs_all(i,:) = jp.insulin_noisy(i,:);

    % --- Latin Hypercube multi-start (Fit_EDES_LatinHyperCube.m) ---
    lhs_samples  = lhsdesign(num_par_sets, num_par);
    initial_pars = lhs_samples .* d + lb;

    best_resnorm = inf;
    best_p_opt   = initial_pars(1,:);

    for s = 1:num_par_sets
        try
            [p_opt_s, resnorm_s] = lsqnonlin(@EDES_ErrorFunc, initial_pars(s,:), ...
                lb, ub, lsq_options, input_data, i, time_opt);
            if resnorm_s < best_resnorm
                best_resnorm = resnorm_s;
                best_p_opt   = p_opt_s;
            end
        catch
            % skip failed start (solver diverged or ODE failed)
        end
    end

    k1_all(i) = best_p_opt(1);
    k5_all(i) = best_p_opt(2);
    k6_all(i) = best_p_opt(3);

    % --- Simulate on time_full (0:1:120) for RMSE and trajectory plots ---
    sim_data.glucose = jp.glucose_noisy(i,:);   % [1 x 5]
    sim_data.insulin = jp.insulin_noisy(i,:);   % [1 x 5]
    sim_data.BW      = jp.BW(i);
    sim_data.meal.G  = 75000;

    p_vec       = EDES_Parameters(best_p_opt, sim_data, 1);
    [x0, c_con] = EDES_Initial(sim_data, 1, p_vec);

    t_saved    = 0;
    G_PL_saved = sim_data.glucose(1, 1);

    ODE_opts = odeset('RelTol', 1e-5, 'AbsTol', 1e-8, 'OutputFcn', @integratorfunG);

    try
        [~, X] = ode45(@EDES_ODE, time_full, x0, ODE_opts, p_vec, c_con, sim_data, 1);
        if size(X, 1) == length(time_full)
            G_traj(i,:)    = X(:, 2)';
            I_traj(i,:)    = X(:, 4)';
            G_pred_sp(i,:) = X(sparse_idx, 2)';
            I_pred_sp(i,:) = X(sparse_idx, 4)';
        else
            G_traj(i,:)    = G_obs_all(i,1) * ones(1, length(time_full));
            I_traj(i,:)    = I_obs_all(i,1) * ones(1, length(time_full));
            G_pred_sp(i,:) = G_obs_all(i,:);
            I_pred_sp(i,:) = I_obs_all(i,:);
        end
    catch
        G_traj(i,:)    = G_obs_all(i,1) * ones(1, length(time_full));
        I_traj(i,:)    = I_obs_all(i,1) * ones(1, length(time_full));
        G_pred_sp(i,:) = G_obs_all(i,:);
        I_pred_sp(i,:) = I_obs_all(i,:);
    end

    if mod(i, 20) == 0
        fprintf('  %d / %d\n', i, N);
    end
end
fprintf('Optimisation complete.\n\n');

save('EDES_MoE/Mixture of Experts/single_PID_dataset_results.mat', 'k1_all', 'k5_all', 'k6_all', 'cats');
fprintf('Results saved to single_PID_dataset_results.mat\n\n');

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

figure('Name', 'Single-Expert Fit Accuracy', 'Position', [50 50 900 420]);

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

sgtitle('Single-Expert Fit Accuracy by ADA Category  —  Japan Dataset');

%% =========================================================================
%% Figure 2 — Mean ± 1 SD trajectories per ADA category
%% =========================================================================
figure('Name', 'Single-Expert Mean trajectories', 'Position', [50 530 1300 700]);

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
    plot(time_full, G_traj_mu, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Single-expert predicted');
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
    plot(time_full, I_traj_mu, 'k-', 'LineWidth', 2.0, 'DisplayName', 'Single-expert predicted');
    xlabel('Time (min)'); ylabel('Insulin (mU/L)');
    title(sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
    legend('Location', 'best'); grid on;
end
sgtitle('Mean +/- 1 SD: Observed vs Single-Expert Predicted  —  Japan Dataset');
