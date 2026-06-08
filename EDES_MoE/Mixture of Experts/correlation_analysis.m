%% correlation_analysis.m
% Compares two correlation analyses for MoE vs single-expert approaches:
%
%   Figure 1 — k5 vs GIR (Glucose Infusion Rate, gold standard for insulin
%              sensitivity):  MoE-fitted k5  |  single-expert-fitted k5
%
%   Figure 2 — k6 vs incr_AUC_IRI_10 (gold standard for insulin production):
%              MoE personalised k6  |  single-expert-fitted k6
%              MoE personalised k6 = sum_e( w_e * k6_e ) using per-patient
%              gating weights stored in MoE_dataset_results.mat.
%
% Prerequisites: MoE_dataset_results.mat, single_PID_dataset_results.mat,
%               japan_population_labelled.mat

clear; clc;

expert_names = {'NGT', 'IGT', 'T2DM'};
colors       = {[0.18 0.63 0.18], [0.93 0.69 0.13], [0.80 0.15 0.15]};

%% -------------------------------------------------------------------------
%% Load optimisation results and gold-standard measurements
%% -------------------------------------------------------------------------
fprintf('Loading MoE results (MoE_dataset_results.mat)...\n');
res_moe = load('MoE_dataset_results.mat');
k5_moe  = res_moe.k5_all;
w_all   = res_moe.w_all;    % [N x 3] per-patient gating weights
cats    = res_moe.cats;
N       = numel(k5_moe);

fprintf('Loading single-expert results (single_PID_dataset_results.mat)...\n');
res_se = load('single_PID_dataset_results.mat');
k5_se  = res_se.k5_all;
k6_se  = res_se.k6_all;

fprintf('Loading japan_population_labelled.mat...\n');
raw = load('japan_population_labelled.mat');
jp  = raw.japan_population;
GIR     = double(jp.GIR(:));
AUC_IRI = double(jp.incr_AUC_IRI_10(:));

%% -------------------------------------------------------------------------
%% MoE personalised k6 = weighted sum of expert k6 values
%% Expert k6 values: NGT=0.079, IGT=0.089, T2DM=0.000  (pids(:,2))
%% -------------------------------------------------------------------------
k6_experts = [0.079; 0.089; 0.000];   % [NGT; IGT; T2DM]
k6_moe     = w_all * k6_experts;      % [N x 1]

%% =========================================================================
%% FIGURE 1 — k5 vs GIR
%% =========================================================================

%% Remove patients with missing values (separately per approach)
valid_moe_k5 = ~isnan(GIR) & ~isnan(k5_moe);
valid_se_k5  = ~isnan(GIR) & ~isnan(k5_se);

GIR_moe_v = GIR(valid_moe_k5);   k5_moe_v = k5_moe(valid_moe_k5);   cats_moe_k5 = cats(valid_moe_k5);
GIR_se_v  = GIR(valid_se_k5);    k5_se_v  = k5_se(valid_se_k5);     cats_se_k5  = cats(valid_se_k5);

fprintf('Patients with valid GIR and k5 — MoE: %d / %d   Single-expert: %d / %d\n\n', ...
    sum(valid_moe_k5), N, sum(valid_se_k5), N);

%% Correlations — k5 vs GIR (MoE)
[r_p_moe_k5, p_p_moe_k5] = corr(k5_moe_v, GIR_moe_v, 'Type', 'Pearson');
[r_s_moe_k5, p_s_moe_k5] = corr(k5_moe_v, GIR_moe_v, 'Type', 'Spearman');

fprintf('=== MoE  k5 vs GIR  (n=%d) ===\n', sum(valid_moe_k5));
fprintf('Pearson  r   = %+.3f   p = %.4f\n', r_p_moe_k5, p_p_moe_k5);
fprintf('Spearman rho = %+.3f   p = %.4f\n', r_s_moe_k5, p_s_moe_k5);
idx_cat_moe_k5 = {find(cats_moe_k5==1), find(cats_moe_k5==2), find(cats_moe_k5==3)};
fprintf('Per-category Spearman:\n');
for c = 1:3
    idx = idx_cat_moe_k5{c};
    if numel(idx) >= 3
        [rs, ps] = corr(k5_moe_v(idx), GIR_moe_v(idx), 'Type', 'Spearman');
        fprintf('  %s (n=%d):  rho = %+.3f   p = %.4f\n', expert_names{c}, numel(idx), rs, ps);
    else
        fprintf('  %s (n=%d):  too few patients for correlation\n', expert_names{c}, numel(idx));
    end
end
fprintf('\n');

%% Correlations — k5 vs GIR (single expert)
[r_p_se_k5, p_p_se_k5] = corr(k5_se_v, GIR_se_v, 'Type', 'Pearson');
[r_s_se_k5, p_s_se_k5] = corr(k5_se_v, GIR_se_v, 'Type', 'Spearman');

fprintf('=== Single-expert  k5 vs GIR  (n=%d) ===\n', sum(valid_se_k5));
fprintf('Pearson  r   = %+.3f   p = %.4f\n', r_p_se_k5, p_p_se_k5);
fprintf('Spearman rho = %+.3f   p = %.4f\n', r_s_se_k5, p_s_se_k5);
idx_cat_se_k5 = {find(cats_se_k5==1), find(cats_se_k5==2), find(cats_se_k5==3)};
fprintf('Per-category Spearman:\n');
for c = 1:3
    idx = idx_cat_se_k5{c};
    if numel(idx) >= 3
        [rs, ps] = corr(k5_se_v(idx), GIR_se_v(idx), 'Type', 'Spearman');
        fprintf('  %s (n=%d):  rho = %+.3f   p = %.4f\n', expert_names{c}, numel(idx), rs, ps);
    else
        fprintf('  %s (n=%d):  too few patients for correlation\n', expert_names{c}, numel(idx));
    end
end
fprintf('\n');

%% Regression lines
p_fit_moe_k5 = polyfit(k5_moe_v, GIR_moe_v, 1);
x_fit_moe_k5 = linspace(min(k5_moe_v)*0.95, max(k5_moe_v)*1.05, 300);
y_fit_moe_k5 = polyval(p_fit_moe_k5, x_fit_moe_k5);

p_fit_se_k5  = polyfit(k5_se_v, GIR_se_v, 1);
x_fit_se_k5  = linspace(min(k5_se_v)*0.95, max(k5_se_v)*1.05, 300);
y_fit_se_k5  = polyval(p_fit_se_k5, x_fit_se_k5);

%% Plot Figure 1
figure('Name', 'GIR vs fitted k5: MoE vs Single-Expert', 'Position', [100 100 1300 530]);

subplot(1, 2, 1);
hold on;
for c = 1:3
    idx = idx_cat_moe_k5{c};
    scatter(k5_moe_v(idx), GIR_moe_v(idx), 60, 'filled', ...
        'MarkerFaceColor', colors{c}, 'MarkerFaceAlpha', 0.75, ...
        'DisplayName', sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
end
plot(x_fit_moe_k5, y_fit_moe_k5, 'k-', 'LineWidth', 1.8, 'DisplayName', 'Linear fit');
text(0.05, 0.95, sprintf('Pearson  r   = %.3f  (p = %.4f)\nSpearman \\rho = %.3f  (p = %.4f)', ...
    r_p_moe_k5, p_p_moe_k5, r_s_moe_k5, p_s_moe_k5), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'BackgroundColor', [1 1 1 0.75], 'EdgeColor', [0.7 0.7 0.7]);
xlabel('Fitted k_5  (min^{-1})', 'FontSize', 12);
ylabel('GIR',                    'FontSize', 12);
title('MoE-fitted k_5 vs GIR',   'FontSize', 13);
legend('Location', 'southeast'); grid on;

subplot(1, 2, 2);
hold on;
for c = 1:3
    idx = idx_cat_se_k5{c};
    scatter(k5_se_v(idx), GIR_se_v(idx), 60, 'filled', ...
        'MarkerFaceColor', colors{c}, 'MarkerFaceAlpha', 0.75, ...
        'DisplayName', sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
end
plot(x_fit_se_k5, y_fit_se_k5, 'k-', 'LineWidth', 1.8, 'DisplayName', 'Linear fit');
text(0.05, 0.95, sprintf('Pearson  r   = %.3f  (p = %.4f)\nSpearman \\rho = %.3f  (p = %.4f)', ...
    r_p_se_k5, p_p_se_k5, r_s_se_k5, p_s_se_k5), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'BackgroundColor', [1 1 1 0.75], 'EdgeColor', [0.7 0.7 0.7]);
xlabel('Fitted k_5  (min^{-1})',         'FontSize', 12);
ylabel('GIR',                            'FontSize', 12);
title('Single-expert-fitted k_5 vs GIR', 'FontSize', 13);
legend('Location', 'southeast'); grid on;

sgtitle('Correlation: fitted k_5 vs gold-standard GIR  —  MoE vs Single-Expert', 'FontSize', 14);

%% =========================================================================
%% FIGURE 2 — k6 vs incr_AUC_IRI_10
%% =========================================================================

%% Remove patients with missing values (separately per approach)
valid_moe_k6 = ~isnan(AUC_IRI) & ~isnan(k6_moe);
valid_se_k6  = ~isnan(AUC_IRI) & ~isnan(k6_se);

AUC_moe_v = AUC_IRI(valid_moe_k6);   k6_moe_v = k6_moe(valid_moe_k6);   cats_moe_k6 = cats(valid_moe_k6);
AUC_se_v  = AUC_IRI(valid_se_k6);    k6_se_v  = k6_se(valid_se_k6);     cats_se_k6  = cats(valid_se_k6);

fprintf('Patients with valid incr_AUC_IRI_10 and k6 — MoE: %d / %d   Single-expert: %d / %d\n\n', ...
    sum(valid_moe_k6), N, sum(valid_se_k6), N);

%% Correlations — k6 vs incr_AUC_IRI_10 (MoE)
[r_p_moe_k6, p_p_moe_k6] = corr(k6_moe_v, AUC_moe_v, 'Type', 'Pearson');
[r_s_moe_k6, p_s_moe_k6] = corr(k6_moe_v, AUC_moe_v, 'Type', 'Spearman');

fprintf('=== MoE  k6 vs incr_AUC_IRI_10  (n=%d) ===\n', sum(valid_moe_k6));
fprintf('Pearson  r   = %+.3f   p = %.4f\n', r_p_moe_k6, p_p_moe_k6);
fprintf('Spearman rho = %+.3f   p = %.4f\n', r_s_moe_k6, p_s_moe_k6);
idx_cat_moe_k6 = {find(cats_moe_k6==1), find(cats_moe_k6==2), find(cats_moe_k6==3)};
fprintf('Per-category Spearman:\n');
for c = 1:3
    idx = idx_cat_moe_k6{c};
    if numel(idx) >= 3
        [rs, ps] = corr(k6_moe_v(idx), AUC_moe_v(idx), 'Type', 'Spearman');
        fprintf('  %s (n=%d):  rho = %+.3f   p = %.4f\n', expert_names{c}, numel(idx), rs, ps);
    else
        fprintf('  %s (n=%d):  too few patients for correlation\n', expert_names{c}, numel(idx));
    end
end
fprintf('\n');

%% Correlations — k6 vs incr_AUC_IRI_10 (single expert)
[r_p_se_k6, p_p_se_k6] = corr(k6_se_v, AUC_se_v, 'Type', 'Pearson');
[r_s_se_k6, p_s_se_k6] = corr(k6_se_v, AUC_se_v, 'Type', 'Spearman');

fprintf('=== Single-expert  k6 vs incr_AUC_IRI_10  (n=%d) ===\n', sum(valid_se_k6));
fprintf('Pearson  r   = %+.3f   p = %.4f\n', r_p_se_k6, p_p_se_k6);
fprintf('Spearman rho = %+.3f   p = %.4f\n', r_s_se_k6, p_s_se_k6);
idx_cat_se_k6 = {find(cats_se_k6==1), find(cats_se_k6==2), find(cats_se_k6==3)};
fprintf('Per-category Spearman:\n');
for c = 1:3
    idx = idx_cat_se_k6{c};
    if numel(idx) >= 3
        [rs, ps] = corr(k6_se_v(idx), AUC_se_v(idx), 'Type', 'Spearman');
        fprintf('  %s (n=%d):  rho = %+.3f   p = %.4f\n', expert_names{c}, numel(idx), rs, ps);
    else
        fprintf('  %s (n=%d):  too few patients for correlation\n', expert_names{c}, numel(idx));
    end
end
fprintf('\n');

%% Regression lines
p_fit_moe_k6 = polyfit(k6_moe_v, AUC_moe_v, 1);
x_fit_moe_k6 = linspace(min(k6_moe_v)*0.95, max(k6_moe_v)*1.05, 300);
y_fit_moe_k6 = polyval(p_fit_moe_k6, x_fit_moe_k6);

p_fit_se_k6  = polyfit(k6_se_v, AUC_se_v, 1);
x_fit_se_k6  = linspace(min(k6_se_v)*0.95, max(k6_se_v)*1.05, 300);
y_fit_se_k6  = polyval(p_fit_se_k6, x_fit_se_k6);

%% Plot Figure 2
figure('Name', 'incr AUC IRI vs fitted k6: MoE vs Single-Expert', 'Position', [150 150 1300 530]);

subplot(1, 2, 1);
hold on;
for c = 1:3
    idx = idx_cat_moe_k6{c};
    scatter(k6_moe_v(idx), AUC_moe_v(idx), 60, 'filled', ...
        'MarkerFaceColor', colors{c}, 'MarkerFaceAlpha', 0.75, ...
        'DisplayName', sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
end
plot(x_fit_moe_k6, y_fit_moe_k6, 'k-', 'LineWidth', 1.8, 'DisplayName', 'Linear fit');
text(0.05, 0.95, sprintf('Pearson  r   = %.3f  (p = %.4f)\nSpearman \\rho = %.3f  (p = %.4f)', ...
    r_p_moe_k6, p_p_moe_k6, r_s_moe_k6, p_s_moe_k6), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'BackgroundColor', [1 1 1 0.75], 'EdgeColor', [0.7 0.7 0.7]);
xlabel('Personalised k_6  (min^{-1})',        'FontSize', 12);
ylabel('Incremental AUC_{IRI,10}',            'FontSize', 12);
title('MoE personalised k_6 vs incr AUC IRI', 'FontSize', 13);
legend('Location', 'best'); grid on;

subplot(1, 2, 2);
hold on;
for c = 1:3
    idx = idx_cat_se_k6{c};
    scatter(k6_se_v(idx), AUC_se_v(idx), 60, 'filled', ...
        'MarkerFaceColor', colors{c}, 'MarkerFaceAlpha', 0.75, ...
        'DisplayName', sprintf('%s  (n=%d)', expert_names{c}, numel(idx)));
end
plot(x_fit_se_k6, y_fit_se_k6, 'k-', 'LineWidth', 1.8, 'DisplayName', 'Linear fit');
text(0.05, 0.95, sprintf('Pearson  r   = %.3f  (p = %.4f)\nSpearman \\rho = %.3f  (p = %.4f)', ...
    r_p_se_k6, p_p_se_k6, r_s_se_k6, p_s_se_k6), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', 10, 'BackgroundColor', [1 1 1 0.75], 'EdgeColor', [0.7 0.7 0.7]);
xlabel('Fitted k_6  (min^{-1})',                       'FontSize', 12);
ylabel('Incremental AUC_{IRI,10}',                     'FontSize', 12);
title('Single-expert-fitted k_6 vs incr AUC IRI',      'FontSize', 13);
legend('Location', 'best'); grid on;

sgtitle('Correlation: fitted k_6 vs incr AUC_{IRI,10}  —  MoE vs Single-Expert', 'FontSize', 14);
