%% Label_VirtualPopulation.m
% Assigns ADA-based glycaemic categories (NGT / IGT / T2DM) to each
% virtual patient in virtual_population.mat, then splits the dataset into
% three category-specific structs and saves everything to
% virtual_population_labelled.mat.
%
% ADA criteria (applied to glucose_noisy):
%   T2DM : fasting >= 7.0 mmol/L  OR  2-h >= 11.1 mmol/L
%   or T2DM : T2DM : fasting >= 7.0 mmol/L  AND  2-h >= 11.1 mmol/L??????
%   IGT  : NOT T2DM  AND  (5.6 <= fasting <= 6.9  OR  7.8 <= 2-h <= 11.1)
%   or IGT : NOT T2DM AND(7.8 <= 2-h <= 11.1)????
%   NGT  : fasting < 5.6  AND  2-h < 7.8
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

%% ========================================================================
% Load virtual population
% =========================================================================
fprintf('Loading virtual_population.mat...\n');
load('virtual_population.mat', 'virtual_population');

time          = virtual_population.time;           % [1 x n_t]
glucose_noisy = virtual_population.glucose_noisy;  % [n_valid x n_t]
insulin_noisy = virtual_population.insulin_noisy;
glucose_clean = virtual_population.glucose_clean;
insulin_clean = virtual_population.insulin_clean;
param_matrix  = virtual_population.param_matrix;
n_valid       = virtual_population.n_valid;

%% ========================================================================
% Extract diagnostic glucose values
% =========================================================================
% Fasting = first time point (t = 0 min)
idx_fast = find(time == 0, 1);
% 2-hour post-load = t = 120 min
idx_2h   = find(time == 120, 1);

if isempty(idx_fast) || isempty(idx_2h)
    error('Could not locate t=0 or t=120 in the time vector.');
end

G_fasting = glucose_noisy(:, idx_fast);   % [n_valid x 1]
G_2h      = glucose_noisy(:, idx_2h);     % [n_valid x 1]

%% ========================================================================
% Apply ADA rules
% Precedence: T2DM > IGT > NGT  (T2DM checked first)
% =========================================================================
%is_T2DM = (G_fasting >= 7.0) | (G_2h >= 11.1);
is_T2DM = (G_fasting >= 7.0) & (G_2h >= 11.1);
%is_IGT  = ~is_T2DM & ((G_fasting >= 5.6 & G_fasting <= 6.9) | ...
 %                      (G_2h      >= 7.8 & G_2h      <= 11.1));
is_IGT = ~is_T2DM & (G_2h >= 7.8 & G_2h <= 11.1);
is_NGT  = (G_fasting < 5.6 & G_2h < 7.8);   % i.e. fasting < 5.6 AND 2-h < 7.8

labels = repmat({'NGT'}, n_valid, 1);
labels(is_IGT)  = {'IGT'};
labels(is_T2DM) = {'T2DM'};

n_NGT  = sum(is_NGT);
n_IGT  = sum(is_IGT);
n_T2DM = sum(is_T2DM);

fprintf('\nADA labelling results (N = %d):\n', n_valid);
fprintf('  NGT  : %4d  (%.1f%%)\n', n_NGT,  100*n_NGT /n_valid);
fprintf('  IGT  : %4d  (%.1f%%)\n', n_IGT,  100*n_IGT /n_valid);
fprintf('  T2DM : %4d  (%.1f%%)\n', n_T2DM, 100*n_T2DM/n_valid);

%% ========================================================================
% Helper: build a sub-dataset struct for a logical index mask
% =========================================================================
make_subset = @(mask, cat_name) struct( ...
    'category',      cat_name, ...
    'time',          time, ...
    'glucose_noisy', glucose_noisy(mask,:), ...
    'insulin_noisy', insulin_noisy(mask,:), ...
    'glucose_clean', glucose_clean(mask,:), ...
    'insulin_clean', insulin_clean(mask,:), ...
    'param_matrix',  param_matrix(mask,:),  ...
    'param_names',   {virtual_population.param_names}, ...
    'G_fasting',     G_fasting(mask),       ...
    'G_2h',          G_2h(mask),            ...
    'labels',        {labels(mask)},        ...
    'n',             sum(mask)              );

dataset_NGT  = make_subset(is_NGT,  'NGT');
dataset_IGT  = make_subset(is_IGT,  'IGT');
dataset_T2DM = make_subset(is_T2DM, 'T2DM');

%% ========================================================================
% Attach labels back to the full population struct
% =========================================================================
virtual_population.labels    = labels;
virtual_population.G_fasting = G_fasting;
virtual_population.G_2h      = G_2h;
virtual_population.is_NGT    = is_NGT;
virtual_population.is_IGT    = is_IGT;
virtual_population.is_T2DM   = is_T2DM;

%% ========================================================================
% Save
% =========================================================================
save('virtual_population_labelled.mat', ...
     'virtual_population', ...
     'dataset_NGT', 'dataset_IGT', 'dataset_T2DM', ...
     '-v7.3');
fprintf('\nSaved: virtual_population_labelled.mat\n');
fprintf('  Fields: virtual_population, dataset_NGT, dataset_IGT, dataset_T2DM\n');

%% ========================================================================
% Summary figure
% =========================================================================
t_plot  = time;
n_show  = 100;   % trajectories per category in the plot

colors = struct('NGT',  [0.18 0.63 0.18], ...   % green
                'IGT',  [0.93 0.69 0.13], ...   % amber
                'T2DM', [0.80 0.15 0.15]);       % red

figure('Name','Virtual Population - ADA Categories','Position',[80 80 1400 550]);

categories = {'NGT','IGT','T2DM'};
datasets   = {dataset_NGT, dataset_IGT, dataset_T2DM};

for col = 1:2   % col 1 = glucose, col 2 = insulin
    for r = 1:3
        subplot(2, 3, (col-1)*3 + r);
        ds  = datasets{r};
        cat = categories{r};
        clr = colors.(cat);

        n_avail = ds.n;
        n_draw  = min(n_show, n_avail);
        idx     = randperm(n_avail, n_draw);

        if col == 1
            traces  = ds.glucose_noisy;
            med_val = median(ds.glucose_noisy, 1);
            ylab    = 'Glucose (mmol/L)';
        else
            traces  = ds.insulin_noisy;
            med_val = median(ds.insulin_noisy, 1);
            ylab    = 'Insulin (uIU/mL)';
        end

        plot(t_plot, traces(idx,:)', 'Color', [clr 0.12], 'LineWidth', 0.5);
        hold on;
        plot(t_plot, med_val, 'Color', clr*0.6, 'LineWidth', 2);
        xlabel('Time (min)'); ylabel(ylab);
        title(sprintf('%s  (n=%d)', cat, n_avail));
        xlim([0 480]); grid on;
    end
end

sgtitle(sprintf('EDES Virtual Population ??? ADA Categories  |  N_{total} = %d', n_valid));
