%% Analyze_Japan_Features.m
% Step 2 of the data-improvement plan: characterise the REAL (Japan) data.
%
% Two products, both feeding step 3/4:
%   A. Per-category MARGINALS of every feature (NGT / IGT / T2DM):
%      mean, sd, median, IQR, 5th/95th pct, n. These ARE the target
%      distributions the synthetic generator must reproduce. The rows
%      flagged [EDES-PRIOR] map directly onto the LHS bounds in
%      Generate_VirtualPopulation.m and tell us exactly how to re-prior.
%   B. Feature -> label SIGNAL: Spearman rho vs the ordinal label
%      (NGT=1, IGT=2, T2DM=3) and the correlation ratio eta^2 (one-way),
%      ranked. Plus the feature-feature Spearman matrix - needed in step 4
%      to sample age/BMI *correlated* with the metabolic parameters
%      instead of independently.
%
% No toolbox functions: Spearman = rank-then-Pearson, implemented locally.
%
% Requires: Convert_Japan_Dataset.m  (-> japan_population_labelled.mat)
% Outputs : japan_feature_analysis.mat (struct `jf`) + 3 figures + tables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

fprintf('Loading japan_population_labelled.mat...\n');
load('japan_population_labelled.mat', 'japan_population');
J = japan_population;
t5 = J.time(:)';                                  % [0 30 60 90 120]

%% ========================================================================
% Build the feature table.
%   col 1 = display name
%   col 2 = numeric vector (n x 1)
%   col 3 = unit
%   col 4 = true if this feature is an EDES generative prior
% =========================================================================
G  = J.glucose_noisy;   I = J.insulin_noisy;      % [n x 5], mmol/L & mU/L
G0 = G(:,1); I0 = I(:,1); G2h = G(:,5); I2h = I(:,5);
Gpeak = max(G,[],2); Ipeak = max(I,[],2);
Gauc  = trapz(t5, G, 2); Iauc = trapz(t5, I, 2);
% HOMA-IR = FPI(uU/mL) * FPG(mmol/L) / 22.5   (uU/mL == mU/L)
HOMA = I0 .* G0 ./ 22.5;

feat = {
%   name                vector            unit          EDES-prior?
   'age',               dbl(J.age),       'years',      false
   'sex (male=1)',      dbl(J.sex),       '-',          false
   'BW',                dbl(J.BW),        'kg',         true
   'BMI',               dbl(J.BMI),       'kg/m^2',     false
   'G_b (fasting gluc)',G0,               'mmol/L',     true
   'I_PL_b (fast ins)', I0,               'mU/L',       true
   'G_2h',              G2h,              'mmol/L',     false
   'I_2h',              I2h,              'mU/L',       false
   'G_peak',            Gpeak,            'mmol/L',     false
   'I_peak',            Ipeak,            'mU/L',       false
   'G_AUC(0-120)',      Gauc,             'mmol/L*min', false
   'I_AUC(0-120)',      Iauc,             'mU/L*min',   false
   'HOMA-IR',           HOMA,             '-',          false
   'Matsuda',           dbl(J.matsuda),   '-',          false
   'QUICKI',            dbl(J.quicki),    '-',          false
   'oral DI',           dbl(J.DI),        '-',          false
   'GIR',               dbl(J.GIR),       '-',          false
};
fname = feat(:,1); fvec = feat(:,2); funit = feat(:,3); fprior = cell2mat(feat(:,4));
nF = numel(fname);

cats   = {'NGT','IGT','T2DM'};
masks  = {J.is_NGT, J.is_IGT, J.is_T2DM};
labcode = nan(numel(G0),1);                       % ordinal label 1/2/3
labcode(J.is_NGT)=1; labcode(J.is_IGT)=2; labcode(J.is_T2DM)=3;

%% ========================================================================
% A. Per-category marginals
% =========================================================================
jf = struct(); jf.categories = cats; jf.features = fname;
fprintf('\n========================================================================\n');
fprintf(' A. JAPAN PER-CATEGORY MARGINALS  (mean +/- sd | [p5  median  p95] | n)\n');
fprintf('    rows tagged [EDES-PRIOR] set the LHS bounds in step 3\n');
fprintf('========================================================================\n');
for f = 1:nF
    x = fvec{f};
    tag = ''; if fprior(f), tag = '  [EDES-PRIOR]'; end
    fprintf('\n%-22s (%s)%s\n', fname{f}, funit{f}, tag);
    for c = 1:3
        xc = x(masks{c}); xc = xc(isfinite(xc));
        if numel(xc) < 2
            fprintf('  %-5s n=%-3d  (insufficient)\n', cats{c}, numel(xc));
            S = struct('n',numel(xc),'mean',NaN,'sd',NaN,'median',NaN, ...
                       'p5',NaN,'p25',NaN,'p75',NaN,'p95',NaN,'min',NaN,'max',NaN);
        else
            S = struct('n',numel(xc),'mean',mean(xc),'sd',std(xc), ...
                'median',median(xc),'p5',prctile(xc,5),'p25',prctile(xc,25), ...
                'p75',prctile(xc,75),'p95',prctile(xc,95),'min',min(xc),'max',max(xc));
            fprintf('  %-5s n=%-3d  %8.2f +/- %7.2f | [%8.2f %8.2f %8.2f]\n', ...
                cats{c}, S.n, S.mean, S.sd, S.p5, S.median, S.p95);
        end
        jf.marginal.(matlab.lang.makeValidName(fname{f})).(cats{c}) = S;
    end
end

%% ========================================================================
% B. Feature -> label signal:  Spearman rho (ordinal) + eta^2 (one-way)
% =========================================================================
rho = nan(nF,1); eta2 = nan(nF,1);
for f = 1:nF
    x = fvec{f}; ok = isfinite(x) & isfinite(labcode);
    if nnz(ok) < 5, continue; end
    rho(f)  = local_spearman(x(ok), labcode(ok));
    eta2(f) = local_eta2(x(ok), labcode(ok));
end
[~, ord] = sort(abs(rho), 'descend', 'MissingPlacement','last');
jf.signal = struct('feature', {fname}, 'spearman_rho_vs_label', rho, ...
                    'eta2_vs_label', eta2);

fprintf('\n========================================================================\n');
fprintf(' B. FEATURE -> LABEL SIGNAL  (ranked by |Spearman rho| vs NGT<IGT<T2DM)\n');
fprintf('    |rho|: <0.1 none, 0.1-0.3 weak, 0.3-0.5 moderate, >0.5 strong\n');
fprintf('========================================================================\n');
fprintf('%-22s | %8s | %8s\n', 'feature', 'rho', 'eta^2');
for r = 1:nF
    f = ord(r);
    fprintf('%-22s | %+8.3f | %8.3f\n', fname{f}, rho(f), eta2(f));
end

%% ========================================================================
% C. Feature-feature Spearman matrix (for correlated sampling in step 4).
%    Restricted to continuous, generation-relevant features.
% =========================================================================
sel_names = {'age','BW','BMI','G_b (fasting gluc)','I_PL_b (fast ins)', ...
             'HOMA-IR','Matsuda','QUICKI','oral DI'};
sel = find(ismember(fname, sel_names));
M = nan(numel(sel));
for a = 1:numel(sel)
    for b = 1:numel(sel)
        xa = fvec{sel(a)}; xb = fvec{sel(b)};
        ok = isfinite(xa) & isfinite(xb);
        if nnz(ok) >= 5, M(a,b) = local_spearman(xa(ok), xb(ok)); end
    end
end
jf.corr_matrix = struct('features', {fname(sel)}, 'spearman', M);

fprintf('\n========================================================================\n');
fprintf(' C. FEATURE-FEATURE SPEARMAN (generation-relevant; used in step 4)\n');
fprintf('========================================================================\n');
fprintf('%-20s', '');
for b = 1:numel(sel), fprintf('%8s', sprintf('f%d',b)); end
fprintf('\n');
for a = 1:numel(sel)
    fprintf('f%-2d %-16s', a, fname{sel(a)});
    for b = 1:numel(sel), fprintf('%8.2f', M(a,b)); end
    fprintf('\n');
end

save('japan_feature_analysis.mat', 'jf');
fprintf('\nSaved: japan_feature_analysis.mat\n');

%% ========================================================================
% Figures
% =========================================================================
clr = [0.18 0.63 0.18; 0.93 0.69 0.13; 0.80 0.15 0.15];   % NGT/IGT/T2DM

% Fig 1: per-category distributions of the EDES-prior features (the targets)
pidx = find(fprior);
figure('Name','Japan EDES-prior feature distributions','Color','w', ...
       'Position',[60 60 360*numel(pidx) 420]);
for j = 1:numel(pidx)
    f = pidx(j); subplot(1, numel(pidx), j); hold on;
    grp = []; pos = [];
    for c = 1:3
        xc = fvec{f}(masks{c}); xc = xc(isfinite(xc));
        bx = (c-1);
        % simple box: median, IQR, 5-95 whisker
        q = prctile(xc,[5 25 50 75 95]);
        fill(bx+[-.3 .3 .3 -.3], q([2 2 4 4]), clr(c,:), 'FaceAlpha',.35,'EdgeColor',clr(c,:));
        plot(bx+[-.3 .3], [q(3) q(3)], '-', 'Color', clr(c,:),'LineWidth',2);
        plot([bx bx], q([1 2]), '-', 'Color', clr(c,:));
        plot([bx bx], q([4 5]), '-', 'Color', clr(c,:));
        jitter = (rand(numel(xc),1)-.5)*.25;
        scatter(bx+jitter, xc, 10, clr(c,:), 'filled', 'MarkerFaceAlpha',.3);
    end
    set(gca,'XTick',0:2,'XTickLabel',cats); grid on;
    title(sprintf('%s (%s)', fname{f}, funit{f}));
end
sgtitle('Japan: per-category distribution of EDES-prior features (step-3 targets)');

% Fig 2: feature -> label signal
figure('Name','Japan feature -> label signal','Color','w','Position',[80 80 720 480]);
barh(abs(rho(flipud(ord)))); set(gca,'YTick',1:nF,'YTickLabel',fname(flipud(ord)));
xlabel('|Spearman \rho|  vs ordinal label (NGT<IGT<T2DM)'); grid on;
title('Which features carry label signal (longer = more discriminative)');

% Fig 3: feature-feature correlation heatmap
figure('Name','Japan feature-feature Spearman','Color','w','Position',[100 100 620 560]);
imagesc(M, [-1 1]); colorbar;
set(gca,'XTick',1:numel(sel),'XTickLabel',fname(sel),'XTickLabelRotation',45, ...
        'YTick',1:numel(sel),'YTickLabel',fname(sel));
for a=1:numel(sel), for b=1:numel(sel)
    if isfinite(M(a,b))
        text(b,a,sprintf('%.2f',M(a,b)),'HorizontalAlignment','center', ...
            'FontSize',8,'Color',(abs(M(a,b))>0.6)*[1 1 1]);
    end
end, end
title('Feature-feature Spearman (drives correlated age/BMI sampling in step 4)');

fprintf('\nDone. Inspect jf.marginal (step-3 targets) and jf.signal (feature relevance).\n');

%% ========================================================================
% Local functions
% =========================================================================
function v = dbl(x)
% Coerce a struct field (possibly cell/char) to a numeric column.
    if iscell(x), x = str2double(x); end
    v = double(x(:));
end

function r = local_spearman(x, y)
% Spearman rho = Pearson on ranks (average ranks for ties).
    rx = local_rank(x(:)); ry = local_rank(y(:));
    rx = rx - mean(rx);    ry = ry - mean(ry);
    denom = sqrt(sum(rx.^2) * sum(ry.^2));
    if denom == 0, r = 0; else, r = sum(rx.*ry) / denom; end
end

function rnk = local_rank(x)
% Average (fractional) ranks, handling ties.
    [xs, ix] = sort(x);
    rnk = zeros(size(x));
    n = numel(x); i = 1;
    base = (1:n)';
    while i <= n
        j = i;
        while j < n && xs(j+1) == xs(i), j = j + 1; end
        rnk(ix(i:j)) = mean(base(i:j));
        i = j + 1;
    end
end

function e2 = local_eta2(x, g)
% Correlation ratio eta^2 = between-group SS / total SS (one-way).
    grand = mean(x); sst = sum((x - grand).^2);
    if sst == 0, e2 = 0; return; end
    ssb = 0; u = unique(g);
    for k = 1:numel(u)
        xk = x(g == u(k));
        ssb = ssb + numel(xk) * (mean(xk) - grand)^2;
    end
    e2 = ssb / sst;
end
