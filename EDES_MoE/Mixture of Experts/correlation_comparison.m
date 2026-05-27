%% correlation_comparison.m
% Compares the correlation between the fitted k5 parameter and gold-standard
% GIR (Glucose Infusion Rate from euglycemic clamp) for two approaches:
%   (1) MoE-fitted k5          (from dataset_results.mat)
%   (2) Single-expert-fitted k5 (from single_expert_results.mat)
%
% Output
%   - Console: Pearson r, Spearman rho, p-values (overall + per category)
%              printed for both approaches side by side
%   - Figure:  1 x 2 subplots — same layout as correlation.m, one per approach
%              Each subplot: scatter coloured by ADA category, linear
%              regression line, and stats annotation
%
% Prerequisites: MoE_dataset_results.mat, single_PID_dataset_results.mat,
%               japan_population_labelled.mat

clear; clc;

expert_names = {'NGT', 'IGT', 'T2DM'};
colors       = {[0.18 0.63 0.18], [0.93 0.69 0.13], [0.80 0.15 0.15]};

%% -------------------------------------------------------------------------
%% Load optimisation results and gold-standard GIR
%% -------------------------------------------------------------------------
fprintf('Loading MoE results (MoE_dataset_results.mat)...\n');
res_moe = load('MoE_dataset_results.mat');
k5_moe  = res_moe.k5_all;
cats    = res_moe.cats;
N       = numel(k5_moe);

fprintf('Loading single-expert results (single_PID_dataset_results.mat)...\n');
res_se = load('single_PID_dataset_results.mat');
k5_se  = res_se.k5_all;

fprintf('Loading japan_population_labelled.mat...\n');
raw = load('japan_population_labelled.mat');
jp  = raw.japan_population;
GIR = double(jp.GIR(:));

%% -------------------------------------------------------------------------
%% Remove patients with missing values (handled separately per approach)
%% -------------------------------------------------------------------------
valid_moe = ~isnan(GIR) & ~isnan(k5_moe);
valid_se  = ~isnan(GIR) & ~isnan(k5_se);

GIR_moe_v  = GIR(valid_moe);   k5_moe_v = k5_moe(valid_moe);   cats_moe = cats(valid_moe);
GIR_se_v   = GIR(valid_se);    k5_se_v  = k5_se(valid_se);     cats_se  = cats(valid_se);

fprintf('Patients with valid GIR and k5 — MoE: %d / %d   Single-expert: %d / %d\n\n', ...
    sum(valid_moe), N, sum(valid_se), N);

%% -------------------------------------------------------------------------
%% Pearson and Spearman correlations — MoE
%% -------------------------------------------------------------------------
[r_p_moe, p_p_moe] = corr(k5_moe_v, GIR_moe_v, 'Type', 'Pearson');
[r_s_moe, p_s_moe] = corr(k5_moe_v, GIR_moe_v, 'Type', 'Spearman');

fprintf('=== MoE  k5 vs GIR  (n=%d) ===\n', sum(valid_moe));
fprintf('Pearson  r   = %+.3f   p = %.4f\n', r_p_moe, p_p_moe);
fprintf('Spearman rho = %+.3f   p = %.4f\n', r_s_moe, p_s_moe);

idx_cat_moe = {find(cats_moe==1), find(cats_moe==2), find(cats_moe==3)};
fprintf('Per-category Spearman:\n');
for c = 1:3
    idx = idx_cat_moe{c};
    if numel(idx) >= 3
        [rs, ps] = corr(k5_moe_v(idx), GIR_moe_v(idx), 'Type', 'Spearman');
        fprintf('  %s (n=%d):  rho = %+.3f   p = %.4f\n', expert_names{c}, numel(idx), rs, ps);
    else
        fprintf('  %s (n=%d):  too few patients for correlation\n', expert_names{c}, numel(idx));
    end
end
fprintf('\n');

%% -------------------------------------------------------------------------
%% Pearson and Spearman correlations — single expert
%% -------------------------------------------------------------------------
[r_p_se, p_p_se] = corr(k5_se_v, GIR_se_v, 'Type', 'Pearson');
[r_s_se, p_s_se] = corr(k5_se_v, GIR_se_v, 'Type', 'Spearman');

fprintf('=== Single-expert  k5 vs GIR  (n=%d) ===\n', sum(valid_se));
fprintf('Pearson  r   = %+.3f   p = %.4f\n', r_p_se, p_p_se);
fprintf('Spearman rho = %+.3f   p = %.4f\n', r_s_se, p_s_se);

idx_cat_se = {find(cats_se==1), find(cats_se==2), find(cats_se==3)};
fprintf('Per-category Spearman:\n');
for c = 1:3
    idx = idx_cat_se{c};
    if numel(idx) >= 3
        [rs, ps] = corr(k5_se_v(idx), GIR_se_v(idx), 'Type', 'Spearman');
        fprintf('  %s (n=%d):  rho = %+.3f   p = %.4f\n', expert_names{c}, numel(idx), rs, ps);
    else
        fprintf('  %s (n=%d):  too few patients for correlation\n', expert_names{c}, numel(idx));
    end
end
fprintf('\n');

%% -------------------------------------------------------------------------
%% Linear regression lines
%% -------------------------------------------------------------------------
p_fit_moe = polyfit(k5_moe_v, GIR_moe_v, 1);
x_fit_moe = linspace(min(k5_moe_v)*0.95, max(k5_moe_v)*1.05, 300);
y_fit_moe = polyval(p_fit_moe, x_fit_moe);

p_fit_se  = polyfit(k5_se_v, GIR_se_v, 1);
x_fit_se  = linspace(min(k5_se_v)*0.95, max(k5_se_v)*1.05, 300);
y_fit_se  = polyval(p_fit_se, x_fit_se);

%% =========================================================================
%% Figure — side-by-side scatter plots (one per approach)
%% =========================================================================
figure('Name', 'GIR vs fitted k5: MoE vs Single-Expert', 'Position', [100 100 1300 530]);

%% --- Left subplot: MoE ---
subplot(1, 2, 1);
hold on;
for c = 1:3
    idx = idx_cat_moe{c};
    scatter(k5_moe_v(idx), GIR_moe_v(idx), 60, 'filled', ...
        'MarkerFaceColor', colors{c}, 'MarkerFaceAlpha', 0.75, ...
        'DisplayName', sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
end
plot(x_fit_moe, y_fit_moe, 'k-', 'LineWidth', 1.8, 'DisplayName', 'Linear fit');
ann_moe = sprintf('Pearson  r   = %.3f  (p = %.4f)\nSpearman \\rho = %.3f  (p = %.4f)', ...
    r_p_moe, p_p_moe, r_s_moe, p_s_moe);
text(0.05, 0.95, ann_moe, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'BackgroundColor', [1 1 1 0.75], 'EdgeColor', [0.7 0.7 0.7]);
xlabel('Fitted k_5  (min^{-1})', 'FontSize', 12);
ylabel('GIR',                    'FontSize', 12);
title('MoE-fitted k_5 vs GIR',   'FontSize', 13);
legend('Location', 'southeast');
grid on;

%% --- Right subplot: single expert ---
subplot(1, 2, 2);
hold on;
for c = 1:3
    idx = idx_cat_se{c};
    scatter(k5_se_v(idx), GIR_se_v(idx), 60, 'filled', ...
        'MarkerFaceColor', colors{c}, 'MarkerFaceAlpha', 0.75, ...
        'DisplayName', sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
end
plot(x_fit_se, y_fit_se, 'k-', 'LineWidth', 1.8, 'DisplayName', 'Linear fit');
ann_se = sprintf('Pearson  r   = %.3f  (p = %.4f)\nSpearman \\rho = %.3f  (p = %.4f)', ...
    r_p_se, p_p_se, r_s_se, p_s_se);
text(0.05, 0.95, ann_se, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'BackgroundColor', [1 1 1 0.75], 'EdgeColor', [0.7 0.7 0.7]);
xlabel('Fitted k_5  (min^{-1})',           'FontSize', 12);
ylabel('GIR',                              'FontSize', 12);
title('Single-expert-fitted k_5 vs GIR',   'FontSize', 13);
legend('Location', 'southeast');
grid on;

sgtitle('Correlation: fitted k_5 vs gold-standard GIR  —  MoE vs Single-Expert', 'FontSize', 14);
