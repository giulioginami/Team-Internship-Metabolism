%% Quantify_Japan_Virtual_Gap.m
% Step 1 of the data-improvement plan: turn the visual Virtual-vs-Japan
% comparison into NUMBERS, so that "we improved the synthetic data" becomes
% provable and the *type* of mismatch (shape vs offset vs spread vs
% category proportions) is identifiable.
%
% For each ADA category (NGT / IGT / T2DM), each variable (glucose,
% insulin) and each of the 5 standard OGTT time points [0 30 60 90 120]:
%   - SMD : standardized mean difference  (Cohen's d, pooled SD)
%           sign = virtual - japan  (positive => virtual higher)
%   - KS  : two-sample Kolmogorov-Smirnov statistic  (shape+location, 0..1)
%   - W1  : 1-D Wasserstein / earth-mover distance   (original units)
%
% Plus per-category distribution gaps on three clinically meaningful
% summary statistics: peak value, 2-h value, and AUC(0-120); and a
% comparison of the NGT/IGT/T2DM category proportions.
%
% Nothing here is fitted or filtered - this is pure measurement. It is the
% baseline that steps 2-4 are scored against.
%
% Requires (run these first):
%   Generate_VirtualPopulation.m -> Label_VirtualPopulation.m
%   Convert_Japan_Dataset.m
% Outputs:
%   japan_virtual_gap_metrics.mat  (struct `gap`)
%   console tables + 3 figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Optional overrides: set `virt_mat` (and/or `jap_mat`) in the base
% workspace before calling to score a different synthetic dataset, e.g.
%   virt_mat = 'virtual_population_v2_labelled.mat'; Quantify_Japan_Virtual_Gap
% Defaults reproduce the original behaviour.
if ~exist('virt_mat','var') || isempty(virt_mat)
    virt_mat = 'virtual_population_labelled.mat';
end
if ~exist('jap_mat','var') || isempty(jap_mat)
    jap_mat = 'japan_population_labelled.mat';
end
clearvars -except virt_mat jap_mat; clc; close all;

%% ========================================================================
% Load datasets
% =========================================================================
fprintf('Loading %s...\n', virt_mat);
load(virt_mat, 'dataset_NGT', 'dataset_IGT', 'dataset_T2DM');

fprintf('Loading %s...\n', jap_mat);
load(jap_mat, 'japan_population');

categories = {'NGT', 'IGT', 'T2DM'};
variables  = {'glucose_noisy', 'insulin_noisy'};
var_label  = {'Glucose (mmol/L)', 'Insulin (mU/L)'};
ogtt_t     = [0 30 60 90 120];          % standard 5-point OGTT [min]
nC = numel(categories); nV = numel(variables); nT = numel(ogtt_t);

ds_virtual = struct('NGT', dataset_NGT, 'IGT', dataset_IGT, 'T2DM', dataset_T2DM);

% Map the virtual high-resolution time grid (0:480) onto the 5 OGTT points.
time_v   = dataset_NGT.time(:)';
[~, idxV] = arrayfun(@(tt) min(abs(time_v - tt)), ogtt_t);   % nearest-sample
if max(abs(time_v(idxV) - ogtt_t)) > 1e-6
    warning('Virtual time grid does not contain exact OGTT points; using nearest samples (max err %.3g min).', ...
        max(abs(time_v(idxV) - ogtt_t)));
end

% Japan category masks
jp_mask = struct('NGT', japan_population.is_NGT, ...
                 'IGT', japan_population.is_IGT, ...
                 'T2DM', japan_population.is_T2DM);

%% ========================================================================
% Per-timepoint metrics: SMD, KS, Wasserstein-1
% =========================================================================
% gap.(var).SMD / .KS / .W1  are  [nT x nC]  matrices (rows = time, cols = category)
gap = struct();
gap.categories = categories;
gap.ogtt_t     = ogtt_t;

for v = 1:nV
    var = variables{v};
    SMD = nan(nT, nC); KS = nan(nT, nC); W1 = nan(nT, nC);
    nV_cat = nan(1, nC); nJ_cat = nan(1, nC);

    for c = 1:nC
        cat = categories{c};
        Xv_full = ds_virtual.(cat).(var);                 % [nv x nTimeFull]
        Xj_full = japan_population.(var)(jp_mask.(cat), :);% [nj x 5]
        nV_cat(c) = size(Xv_full, 1);
        nJ_cat(c) = size(Xj_full, 1);
        if nV_cat(c) < 3 || nJ_cat(c) < 3
            continue   % too few to compare; leave NaN
        end
        for k = 1:nT
            xv = Xv_full(:, idxV(k));
            xj = Xj_full(:, k);
            xv = xv(isfinite(xv));  xj = xj(isfinite(xj));
            if numel(xv) < 3 || numel(xj) < 3, continue; end
            SMD(k, c) = local_smd(xv, xj);    % virtual - japan
            KS(k, c)  = local_ks(xv, xj);
            W1(k, c)  = local_w1(xv, xj);
        end
    end
    gap.(matlab.lang.makeValidName(var)).SMD = SMD;
    gap.(matlab.lang.makeValidName(var)).KS  = KS;
    gap.(matlab.lang.makeValidName(var)).W1  = W1;
    gap.(matlab.lang.makeValidName(var)).n_virtual = nV_cat;
    gap.(matlab.lang.makeValidName(var)).n_japan   = nJ_cat;
end

%% ========================================================================
% Console report: per-timepoint
% =========================================================================
fprintf('\n========================================================================\n');
fprintf(' PER-TIMEPOINT GAP  (SMD signed virtual-japan | KS 0..1 | W1 in units)\n');
fprintf(' |SMD|: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large\n');
fprintf('========================================================================\n');
for v = 1:nV
    var = matlab.lang.makeValidName(variables{v});
    fprintf('\n--- %s ---\n', var_label{v});
    fprintf('%-6s', 't[min]');
    for c = 1:nC, fprintf(' | %-22s', sprintf('%s (nv=%d, nj=%d)', categories{c}, ...
            gap.(var).n_virtual(c), gap.(var).n_japan(c))); end
    fprintf('\n');
    for k = 1:nT
        fprintf('%-6d', ogtt_t(k));
        for c = 1:nC
            fprintf(' | SMD%+5.2f KS%4.2f W1%6.2f', ...
                gap.(var).SMD(k,c), gap.(var).KS(k,c), gap.(var).W1(k,c));
        end
        fprintf('\n');
    end
end

%% ========================================================================
% Summary statistics per individual, then per-category distribution gap
%   peak  = max over the 5 OGTT points
%   v2h   = value at t = 120
%   AUC   = trapezoid over [0 120] (total area, mmol/L*min or mU/L*min)
% =========================================================================
stat_names = {'peak', 'v2h', 'AUC'};
nS = numel(stat_names);
fprintf('\n========================================================================\n');
fprintf(' SUMMARY-STAT GAP per category   (SMD signed virtual-japan | W1 units)\n');
fprintf('========================================================================\n');

gap.summary = struct();
for v = 1:nV
    var = variables{v};
    vkey = matlab.lang.makeValidName(var);
    fprintf('\n--- %s ---\n', var_label{v});
    fprintf('%-6s', 'cat');
    for s = 1:nS, fprintf(' | %-18s', stat_names{s}); end
    fprintf('\n');
    for c = 1:nC
        cat = categories{c};
        Xv = ds_virtual.(cat).(var)(:, idxV);             % [nv x 5]
        Xj = japan_population.(var)(jp_mask.(cat), :);     % [nj x 5]
        fprintf('%-6s', cat);
        for s = 1:nS
            sv = local_sumstat(Xv, ogtt_t, stat_names{s});
            sj = local_sumstat(Xj, ogtt_t, stat_names{s});
            sv = sv(isfinite(sv)); sj = sj(isfinite(sj));
            if numel(sv) < 3 || numel(sj) < 3
                smd_s = NaN; w1_s = NaN;
            else
                smd_s = local_smd(sv, sj);
                w1_s  = local_w1(sv, sj);
            end
            gap.summary.(vkey).(cat).(stat_names{s}) = ...
                struct('SMD', smd_s, 'W1', w1_s, ...
                       'mean_virtual', mean(sv), 'mean_japan', mean(sj));
            fprintf(' | SMD%+5.2f W1%7.2f', smd_s, w1_s);
        end
        fprintf('\n');
    end
end

%% ========================================================================
% Category proportions  (hidden by renms' per-category plot, but it
% directly affects any classifier trained on the synthetic data)
% =========================================================================
nv_tot = dataset_NGT.n + dataset_IGT.n + dataset_T2DM.n;
prop_v = [dataset_NGT.n, dataset_IGT.n, dataset_T2DM.n] / nv_tot;
nj_each = [sum(japan_population.is_NGT), sum(japan_population.is_IGT), sum(japan_population.is_T2DM)];
prop_j  = nj_each / sum(nj_each);

gap.proportions = struct('categories', {categories}, ...
    'virtual', prop_v, 'japan', prop_j, ...
    'n_virtual', [dataset_NGT.n, dataset_IGT.n, dataset_T2DM.n], 'n_japan', nj_each);

fprintf('\n========================================================================\n');
fprintf(' CATEGORY PROPORTIONS\n');
fprintf('========================================================================\n');
fprintf('%-6s | %-18s | %-18s | %-8s\n', 'cat', 'virtual', 'japan', 'abs diff');
for c = 1:nC
    fprintf('%-6s | %6.1f%% (n=%5d) | %6.1f%% (n=%4d) | %+6.1f pp\n', ...
        categories{c}, 100*prop_v(c), gap.proportions.n_virtual(c), ...
        100*prop_j(c), nj_each(c), 100*(prop_v(c)-prop_j(c)));
end

%% ========================================================================
% Single headline number: mean |SMD| over all timepoints/categories/vars.
% This is the scalar that steps 2-4 must drive DOWN to claim improvement.
% =========================================================================
allSMD = [];
for v = 1:nV
    allSMD = [allSMD; gap.(matlab.lang.makeValidName(variables{v})).SMD(:)]; %#ok<AGROW>
end
gap.headline_mean_absSMD = mean(abs(allSMD), 'omitnan');
fprintf('\n>> HEADLINE  mean|SMD| (all timepoints/categories/vars) = %.3f\n', ...
    gap.headline_mean_absSMD);
fprintf('   (lower is better; this is the baseline to beat in steps 2-4)\n');

save('japan_virtual_gap_metrics.mat', 'gap');
fprintf('\nSaved: japan_virtual_gap_metrics.mat\n');

%% ========================================================================
% Figures
% =========================================================================
% Fig 1: SMD heatmaps (signed) - diverging colormap centred on 0
figure('Name','Gap: signed SMD (virtual - japan)','Color','w','Position',[80 80 1100 460]);
for v = 1:nV
    subplot(1, nV, v);
    M = gap.(matlab.lang.makeValidName(variables{v})).SMD;
    imagesc(M, [-2 2]); colormap(local_diverging()); colorbar;
    set(gca,'XTick',1:nC,'XTickLabel',categories,'YTick',1:nT,'YTickLabel',ogtt_t);
    xlabel('Category'); ylabel('OGTT time (min)');
    title(sprintf('%s  -  signed SMD', var_label{v}));
    for k = 1:nT, for c = 1:nC
        if isfinite(M(k,c))
            text(c, k, sprintf('%+.2f', M(k,c)), 'HorizontalAlignment','center', ...
                'FontSize', 9, 'FontWeight','bold');
        end
    end, end
end
sgtitle('Standardized mean difference  (red = virtual too high, blue = too low)');

% Fig 2: KS heatmaps (0..1, higher = worse)
figure('Name','Gap: KS statistic','Color','w','Position',[100 100 1100 460]);
for v = 1:nV
    subplot(1, nV, v);
    M = gap.(matlab.lang.makeValidName(variables{v})).KS;
    imagesc(M, [0 1]); colormap(gca, flipud(gray)); colorbar;
    set(gca,'XTick',1:nC,'XTickLabel',categories,'YTick',1:nT,'YTickLabel',ogtt_t);
    xlabel('Category'); ylabel('OGTT time (min)');
    title(sprintf('%s  -  KS statistic', var_label{v}));
    for k = 1:nT, for c = 1:nC
        if isfinite(M(k,c))
            text(c, k, sprintf('%.2f', M(k,c)), 'HorizontalAlignment','center', ...
                'FontSize', 9, 'FontWeight','bold', ...
                'Color', (M(k,c)>0.5)*[1 1 1]);
        end
    end, end
end
sgtitle('Two-sample KS distance  (0 = identical, 1 = disjoint)');

% Fig 3: category proportions
figure('Name','Gap: category proportions','Color','w','Position',[120 120 560 420]);
bar(100*[prop_v(:), prop_j(:)]);
set(gca,'XTickLabel',categories); ylabel('% of population');
legend({'Virtual','Japan'},'Location','best'); grid on;
title('ADA category proportions: Virtual vs Japan');

fprintf('\nDone. Use gap.headline_mean_absSMD as the scalar to beat.\n');

%% ========================================================================
% Local functions
% =========================================================================
function d = local_smd(xv, xj)
% Cohen's d with pooled SD; sign = virtual - japan.
    nv = numel(xv); nj = numel(xj);
    sp = sqrt(((nv-1)*var(xv) + (nj-1)*var(xj)) / max(nv+nj-2, 1));
    if sp == 0, d = 0; else, d = (mean(xv) - mean(xj)) / sp; end
end

function ks = local_ks(xv, xj)
% Two-sample Kolmogorov-Smirnov statistic (max ECDF gap), no toolbox.
    xs = sort([xv(:); xj(:)]);
    Fv = local_ecdf(xv, xs);
    Fj = local_ecdf(xj, xs);
    ks = max(abs(Fv - Fj));
end

function F = local_ecdf(x, q)
% Empirical CDF of sample x evaluated at query points q.
    xs = sort(x(:));
    F  = arrayfun(@(qq) sum(xs <= qq), q(:)) / numel(xs);
end

function w = local_w1(xv, xj)
% 1-D Wasserstein-1 (earth mover) distance, original units.
% W1 = integral |F_v^-1(u) - F_j^-1(u)| du, via common quantile grid.
    u  = (0.5:1:999.5) / 1000;            % 1000 mid-quantiles
    qv = quantile(xv, u);
    qj = quantile(xj, u);
    w  = mean(abs(qv - qj));
end

function s = local_sumstat(X5, t5, name)
% Per-row summary statistic from the 5-point OGTT matrix X5 [n x 5].
    switch name
        case 'peak', s = max(X5, [], 2);
        case 'v2h',  s = X5(:, end);
        case 'AUC',  s = trapz(t5, X5, 2);     % total area under curve
        otherwise,   error('unknown stat %s', name);
    end
end

function cmap = local_diverging()
% Blue-white-red diverging colormap (no toolbox dependency).
% Odd length so the exact midpoint is white (m integer).
    m = 127;                       % -> 2*m+1 = 255 entries
    r = [linspace(0.23,1,m+1), ones(1,m)];
    g = [linspace(0.30,1,m+1), linspace(1,0.15,m)];
    b = [ones(1,m+1), linspace(1,0.15,m)];
    cmap = [r(:), g(:), b(:)];
end
