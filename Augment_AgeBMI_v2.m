%% Augment_AgeBMI_v2.m
% STEP 4 of the data-improvement plan: add `age` and `BMI` to the v2
% synthetic population as SAMPLED COVARIATES.
%
% Rationale (from step 2): age is the 5th-strongest label discriminator
% (Spearman rho = +0.71; NGT~30 -> IGT~42 -> T2DM~55) and BMI is moderate
% (+0.52). Neither has a mechanistic hook in EDES, so they are NOT fed
% back into the ODE - they are appended post-hoc. The key requirement is
% that they are sampled JOINTLY with the metabolic state, not
% independently, so virtual patients stay physiologically coherent
% (step-2 showed age<->G_b = +0.49, BMI<->BW = +0.86, BMI<->I_PL_b = +0.41).
%
% Method: per ADA category, fit a 5-D Gaussian to the Japan vector
%   [age, BMI, G_b, I_PL_b, BW]   (G_b = fasting glucose, I_PL_b = fasting
% insulin) and draw (age, BMI) for each virtual individual from the
% CONDITIONAL distribution given that individual's already-sampled
% (G_b, I_PL_b, BW). Small-n categories (IGT) get shrinkage toward the
% pooled covariance for stability. This reproduces both the per-category
% location and the within-category correlations by construction.
%
% Operates on the balanced v2 labelled set; does NOT re-run the ODE.
% Output: virtual_population_v2_aug_labelled.mat  (+ validation figure)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;
rng(42);

%% ========================================================================
% Load v2 synthetic (balanced) + Japan
% =========================================================================
fprintf('Loading virtual_population_v2_labelled.mat...\n');
S = load('virtual_population_v2_labelled.mat', ...
         'virtual_population','dataset_NGT','dataset_IGT','dataset_T2DM');
fprintf('Loading japan_population_labelled.mat...\n');
load('japan_population_labelled.mat', 'japan_population'); J = japan_population;

cats   = {'NGT','IGT','T2DM'};
jmask  = {J.is_NGT, J.is_IGT, J.is_T2DM};
ds     = {S.dataset_NGT, S.dataset_IGT, S.dataset_T2DM};

% Column indices of the "known" block inside param_matrix
pn = S.dataset_NGT.param_names;
iGb = find(strcmp(pn,'G_b')); iIb = find(strcmp(pn,'I_PL_b')); iBW = find(strcmp(pn,'BW'));

% Japan known block
Gb_j_all = J.glucose_noisy(:,1);
Ib_j_all = J.insulin_noisy(:,1);
BW_j_all = double(J.BW(:));
age_j_all = double(J.age(:));
BMI_j_all = double(J.BMI(:));

%% ========================================================================
% Pooled (all-category) Japan stats for shrinkage of small categories
% Variable order everywhere:  [age, BMI, G_b, I_PL_b, BW]
% =========================================================================
Ypool = [age_j_all, BMI_j_all, Gb_j_all, Ib_j_all, BW_j_all];
Ypool = Ypool(all(isfinite(Ypool),2), :);
mu_pool = mean(Ypool,1); S_pool = cov(Ypool);

uIdx = [1 2];    % unknown: age, BMI
kIdx = [3 4 5];  % known:   G_b, I_PL_b, BW
KAPPA = 20;      % shrinkage strength (lambda = n/(n+KAPPA))

%% ========================================================================
% Sample (age, BMI) for each category from the conditional Gaussian
% =========================================================================
fprintf('\nConditional age/BMI sampling per category:\n');
for c = 1:3
    % --- Japan stats for this category (with shrinkage) ---
    Yj = [age_j_all(jmask{c}), BMI_j_all(jmask{c}), Gb_j_all(jmask{c}), ...
          Ib_j_all(jmask{c}), BW_j_all(jmask{c})];
    Yj = Yj(all(isfinite(Yj),2), :);
    n_jc = size(Yj,1);
    mu_c = mean(Yj,1);
    S_c  = cov(Yj);
    lam  = n_jc / (n_jc + KAPPA);
    mu   = lam*mu_c + (1-lam)*mu_pool;
    Sig  = lam*S_c  + (1-lam)*S_pool;

    % --- Conditional moments  p(age,BMI | G_b,I_PL_b,BW) ---
    mu_u = mu(uIdx)';            % 2x1
    mu_k = mu(kIdx)';            % 3x1
    S_uu = Sig(uIdx,uIdx);
    S_uk = Sig(uIdx,kIdx);
    S_kk = Sig(kIdx,kIdx);
    S_kk = S_kk + 1e-9*eye(3);   % numerical guard
    B    = S_uk / S_kk;          % 2x3 regression matrix
    C    = S_uu - B*S_uk';       % conditional covariance (shared in category)
    C    = (C + C')/2;           % symmetrize
    [L,p] = chol(C,'lower');
    if p ~= 0                    % not PD after shrinkage -> diagonal fallback
        L = diag(sqrt(max(diag(C), 1e-6)));
    end

    % --- Draw for every virtual individual in this category ---
    Xk = ds{c}.param_matrix(:, [iGb iIb iBW]);   % [n x 3] known block
    n_v = size(Xk,1);
    Z   = randn(2, n_v);
    M   = mu_u + B*(Xk' - mu_k);                 % 2 x n  conditional means
    AB  = M + L*Z;                               % 2 x n  samples
    age = AB(1,:)'; bmi = AB(2,:)';

    % Physiological clamps
    age = min(max(age, 18), 85);
    bmi = min(max(bmi, 15), 45);

    ds{c}.age = age;
    ds{c}.BMI = bmi;

    fprintf(['  %-4s  n_jp=%2d lambda=%.2f | virtual age %.1f+/-%.1f (Japan %.1f+/-%.1f)' ...
             ' | BMI %.1f+/-%.1f (Japan %.1f+/-%.1f)\n'], cats{c}, n_jc, lam, ...
        mean(age),std(age), mean(Yj(:,1)),std(Yj(:,1)), ...
        mean(bmi),std(bmi), mean(Yj(:,2)),std(Yj(:,2)));
end
dataset_NGT = ds{1}; dataset_IGT = ds{2}; dataset_T2DM = ds{3};

%% ========================================================================
% Attach to the flat virtual_population struct (same row order as labels)
% =========================================================================
virtual_population = S.virtual_population;
virtual_population.age = [dataset_NGT.age; dataset_IGT.age; dataset_T2DM.age];
virtual_population.BMI = [dataset_NGT.BMI; dataset_IGT.BMI; dataset_T2DM.BMI];
virtual_population.version = 'v2_japan_repriored_balanced_ageBMI';

save('virtual_population_v2_aug_labelled.mat', ...
     'virtual_population','dataset_NGT','dataset_IGT','dataset_T2DM','-v7.3');
fprintf('\nSaved: virtual_population_v2_aug_labelled.mat\n');

%% ========================================================================
% Validation: did we reproduce the step-2 signal & correlations?
% =========================================================================
vp = virtual_population;
labcode_v = nan(numel(vp.age),1);
labcode_v(vp.is_NGT)=1; labcode_v(vp.is_IGT)=2; labcode_v(vp.is_T2DM)=3;
labcode_j = nan(numel(age_j_all),1);
labcode_j(J.is_NGT)=1; labcode_j(J.is_IGT)=2; labcode_j(J.is_T2DM)=3;

Gb_v = vp.param_matrix(:, iGb);
fprintf('\n--- Correlation reproduction (virtual vs Japan target) ---\n');
fprintf('  age  -> label : %+.3f  (Japan %+.3f, target +0.71)\n', ...
    sp(vp.age, labcode_v), sp(age_j_all, labcode_j));
fprintf('  BMI  -> label : %+.3f  (Japan %+.3f, target +0.52)\n', ...
    sp(vp.BMI, labcode_v), sp(BMI_j_all, labcode_j));
fprintf('  age  <-> G_b  : %+.3f  (Japan %+.3f, target +0.49)\n', ...
    sp(vp.age, Gb_v), sp(age_j_all, Gb_j_all));
fprintf('  BMI  <-> BW   : %+.3f  (Japan %+.3f, target +0.86)\n', ...
    sp(vp.BMI, vp.param_matrix(:,iBW)), sp(BMI_j_all, BW_j_all));

% Figure: per-category age & BMI, virtual vs Japan
clr = [0.18 0.63 0.18; 0.93 0.69 0.13; 0.80 0.15 0.15];
figure('Name','Step 4: age/BMI virtual vs Japan','Color','w','Position',[80 80 1100 460]);
feat = {'age', vp.age, age_j_all, 'Age (years)'; 'BMI', vp.BMI, BMI_j_all, 'BMI (kg/m^2)'};
for f = 1:2
    subplot(1,2,f); hold on;
    for c = 1:3
        xv = feat{f,2}; xv = xv(eval(sprintf('vp.is_%s',cats{c})));
        xj = feat{f,3}; xj = xj(jmask{c}); xj = xj(isfinite(xj));
        bx = (c-1)*3;
        boxlike(bx,   xv, clr(c,:));            % virtual (filled)
        boxlike(bx+1, xj, clr(c,:)*0.45);       % japan   (dark)
    end
    set(gca,'XTick',(0:2)*3+0.5,'XTickLabel',cats); grid on;
    ylabel(feat{f,4}); title([feat{f,4} '  (left=virtual, right=Japan)']);
end
sgtitle('Step 4 validation: synthetic age/BMI now match Japan per category');

fprintf(['\nNext (task D-4 figure):\n' ...
   '  Compare_Prev_New_Japan        %% REAL vs PREVIOUS vs NEW synthetic\n']);

%% ========================================================================
% Local functions
% =========================================================================
function r = sp(x,y)
    ok = isfinite(x) & isfinite(y); x=x(ok); y=y(ok);
    rx = tr(x); ry = tr(y); rx=rx-mean(rx); ry=ry-mean(ry);
    d = sqrt(sum(rx.^2)*sum(ry.^2)); if d==0, r=0; else, r=sum(rx.*ry)/d; end
end
function rnk = tr(x)
    [xs,ix]=sort(x(:)); n=numel(x); rnk=zeros(n,1); b=(1:n)'; i=1;
    while i<=n
        j=i; while j<n && xs(j+1)==xs(i), j=j+1; end
        rnk(ix(i:j))=mean(b(i:j)); i=j+1;
    end
end
function boxlike(xpos, x, col)
    if numel(x)<2, return; end
    q = prctile(x,[5 25 50 75 95]);
    fill(xpos+[-.35 .35 .35 -.35], q([2 2 4 4]), col, 'FaceAlpha',.4,'EdgeColor',col);
    plot(xpos+[-.35 .35],[q(3) q(3)],'-','Color',col*0.6,'LineWidth',2);
    plot([xpos xpos],q([1 2]),'-','Color',col); plot([xpos xpos],q([4 5]),'-','Color',col);
end
