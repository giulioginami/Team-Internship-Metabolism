%% Label_VirtualPopulation_v2.m
% STEP 3 (cont.). Labels the re-prioredd virtual population
% (virtual_population_v2.mat) using the corrected ADA-OR classifier
% (Classify_Diabetes_2H_OGTT_v2), then STRATIFIED-DOWNSAMPLES to a
% balanced 33/33/33 NGT/IGT/T2DM mix (user decision) so the synthetic
% training set does not inherit the NGT-heavy emergent prior.
%
% Output struct shape is identical to Label_VirtualPopulation.m so that
% Quantify_Japan_Virtual_Gap.m and Compare_Japan_Virtual_Data.m work
% unchanged - just point them at virtual_population_v2_labelled.mat.
%
% Originals (virtual_population_labelled.mat) are untouched -> step 4 can
% compare PREVIOUS vs NEW.
% Output: virtual_population_v2_labelled.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;
rng(42); % reproducible downsampling

%% ========================================================================
% Load re-prioredd virtual population
% =========================================================================
fprintf('Loading virtual_population_v2.mat...\n');
load('virtual_population_v2.mat', 'virtual_population');

time          = virtual_population.time;
glucose_noisy = virtual_population.glucose_noisy;
insulin_noisy = virtual_population.insulin_noisy;
glucose_clean = virtual_population.glucose_clean;
insulin_clean = virtual_population.insulin_clean;
param_matrix  = virtual_population.param_matrix;
n_valid       = virtual_population.n_valid;

%% ========================================================================
% Extract 5-point OGTT diagnostic values and classify (ADA-OR v2)
% =========================================================================
idx_fast = find(time == 0,   1);
idx_30   = find(time == 30,  1);
idx_60   = find(time == 60,  1);
idx_90   = find(time == 90,  1);
idx_120  = find(time == 120, 1);
ogtt_test_glucose = glucose_noisy(:, [idx_fast, idx_30, idx_60, idx_90, idx_120]);
ogtt_test_insulin = insulin_noisy(:, [idx_fast, idx_30, idx_60, idx_90, idx_120]);
G_fasting = glucose_noisy(:, idx_fast);
I_fasting = insulin_noisy(:, idx_fast);
G_2h      = glucose_noisy(:, idx_120);

[is_NGT, is_IGT, is_T2DM, ~] = Classify_Diabetes_2H_OGTT_v2(ogtt_test_glucose, ogtt_test_insulin);
matsuda = Calculate_Matsuda_5_OGTT(ogtt_test_glucose, ogtt_test_insulin);
quicki  = Calculate_QUICKI(G_fasting, I_fasting);

n_NGT_raw  = sum(is_NGT);
n_IGT_raw  = sum(is_IGT);
n_T2DM_raw = sum(is_T2DM);

fprintf('\nADA-OR labelling (raw, before balancing) | N_valid = %d:\n', n_valid);
fprintf('  NGT  : %4d  (%.1f%%)\n', n_NGT_raw,  100*n_NGT_raw /n_valid);
fprintf('  IGT  : %4d  (%.1f%%)\n', n_IGT_raw,  100*n_IGT_raw /n_valid);
fprintf('  T2DM : %4d  (%.1f%%)\n', n_T2DM_raw, 100*n_T2DM_raw/n_valid);

%% ========================================================================
% Stratified downsample -> balanced 33/33/33
% =========================================================================
idx_NGT  = find(is_NGT);
idx_IGT  = find(is_IGT);
idx_T2DM = find(is_T2DM);

n_target = min([n_NGT_raw, n_IGT_raw, n_T2DM_raw]);
if n_target < 1
    error(['A class is empty after ADA-OR labelling (NGT=%d IGT=%d T2DM=%d). ' ...
           'Increase N in Generate_VirtualPopulation_v2.m or widen a kinetic ' ...
           'prior.'], n_NGT_raw, n_IGT_raw, n_T2DM_raw);
end

sel_NGT  = idx_NGT( randsample(numel(idx_NGT),  n_target));
sel_IGT  = idx_IGT( randsample(numel(idx_IGT),  n_target));
sel_T2DM = idx_T2DM(randsample(numel(idx_T2DM), n_target));

fprintf('\nBalanced to 33/33/33: %d per class (limiting class had %d).\n', ...
        n_target, n_target);
if n_target < 100
    warning(['Only %d individuals per balanced class. Consider raising N ' ...
             'in Generate_VirtualPopulation_v2.m for a larger training set.'], n_target);
end

%% ========================================================================
% Helper: build a sub-dataset struct (same shape as original Label)
% =========================================================================
make_subset = @(sel, cat_name) struct( ...
    'category',      cat_name, ...
    'time',          time, ...
    'glucose_noisy', glucose_noisy(sel,:), ...
    'insulin_noisy', insulin_noisy(sel,:), ...
    'glucose_clean', glucose_clean(sel,:), ...
    'insulin_clean', insulin_clean(sel,:), ...
    'param_matrix',  param_matrix(sel,:),  ...
    'param_names',   {virtual_population.param_names}, ...
    'G_fasting',     G_fasting(sel),       ...
    'G_2h',          G_2h(sel),            ...
    'n',             numel(sel)            );

dataset_NGT  = make_subset(sel_NGT,  'NGT');
dataset_IGT  = make_subset(sel_IGT,  'IGT');
dataset_T2DM = make_subset(sel_T2DM, 'T2DM');

%% ========================================================================
% Rebuild a BALANCED virtual_population struct (only selected individuals)
% =========================================================================
sel_all = [sel_NGT; sel_IGT; sel_T2DM];
labels  = [repmat({'NGT'}, n_target,1); repmat({'IGT'}, n_target,1); ...
           repmat({'T2DM'}, n_target,1)];

vp = struct();
vp.time          = time;
vp.glucose_noisy = glucose_noisy(sel_all,:);
vp.insulin_noisy = insulin_noisy(sel_all,:);
vp.glucose_clean = glucose_clean(sel_all,:);
vp.insulin_clean = insulin_clean(sel_all,:);
vp.param_matrix  = param_matrix(sel_all,:);
vp.param_names   = virtual_population.param_names;
vp.G_fasting     = G_fasting(sel_all);
vp.G_2h          = G_2h(sel_all);
vp.labels        = labels;
vp.is_NGT        = [true(n_target,1);  false(n_target,1); false(n_target,1)];
vp.is_IGT        = [false(n_target,1); true(n_target,1);  false(n_target,1)];
vp.is_T2DM       = [false(n_target,1); false(n_target,1); true(n_target,1)];
vp.n_valid       = 3*n_target;
vp.N_attempted   = virtual_population.N_attempted;
vp.version       = 'v2_japan_repriored_balanced';
vp.balance_info  = struct('raw_NGT',n_NGT_raw,'raw_IGT',n_IGT_raw, ...
                          'raw_T2DM',n_T2DM_raw,'per_class',n_target, ...
                          'classifier','Classify_Diabetes_2H_OGTT_v2 (ADA-OR)');
virtual_population = vp;

save('virtual_population_v2_labelled.mat', ...
     'virtual_population', 'dataset_NGT', 'dataset_IGT', 'dataset_T2DM', '-v7.3');
fprintf('\nSaved: virtual_population_v2_labelled.mat (balanced, %d total)\n', 3*n_target);

%% ========================================================================
% Sanity figure: balanced category curves
% =========================================================================
colors = struct('NGT',[0.18 0.63 0.18],'IGT',[0.93 0.69 0.13],'T2DM',[0.80 0.15 0.15]);
categories = {'NGT','IGT','T2DM'};
datasets   = {dataset_NGT, dataset_IGT, dataset_T2DM};

figure('Name','Virtual Population v2 - balanced ADA categories', ...
       'Position',[80 80 1400 550], 'Color','w');
for col = 1:2
    for r = 1:3
        ax = subplot(2,3,(col-1)*3 + r);
        ds = datasets{r}; cat = categories{r}; clr = colors.(cat);
        if col == 1, D = ds.glucose_noisy; ylab='Glucose (mmol/L)';
        else,        D = ds.insulin_noisy; ylab='Insulin (mU/L)'; end
        med = median(D,1); q1 = prctile(D,25,1); q3 = prctile(D,75,1);
        fill([time fliplr(time)], [q1 fliplr(q3)], clr, 'EdgeColor','none', ...
             'FaceAlpha',0.3); hold(ax,'on');
        plot(ax, time, med, '-', 'Color', clr*0.7, 'LineWidth', 2);
        xlabel(ax,'Time (min)'); ylabel(ax,ylab);
        title(ax, sprintf('%s (n=%d)', cat, ds.n)); xlim(ax,[0 180]); grid(ax,'on');
    end
end
sgtitle(sprintf('Virtual v2 (Japan-repriored, ADA-OR, balanced) | %d/class', n_target));

fprintf('\nNext: score it ->  virt_mat=''virtual_population_v2_labelled.mat''; Quantify_Japan_Virtual_Gap\n');
