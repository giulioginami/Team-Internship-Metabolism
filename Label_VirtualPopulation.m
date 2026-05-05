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
idx_30 = find(time == 30, 1);
idx_60 = find(time == 60, 1);
idx_90 = find(time == 90, 1);
idx_120 = find(time == 120, 1);
ogtt_test_glucose = glucose_noisy(:, [idx_fast, idx_30, idx_60, idx_90, idx_120]);
ogtt_test_insulin = insulin_noisy(:, [idx_fast, idx_30, idx_60, idx_90, idx_120]);
G_fasting = glucose_noisy(:, idx_fast);   % [n_valid x 1]
G_2h      = glucose_noisy(:, idx_120);    % [n_valid x 1]
% Apply ADA rules
[is_NGT, is_IGT, is_T2DM, err] = Classify_Diabetes_2H_OGTT(ogtt_test_glucose, ogtt_test_insulin);


%% ========================================================================
% Extract diagnostic glucose values
% =========================================================================
% Fasting = first time point (t = 0 min)
idx_fast = find(time == 0, 1);
idx_30 = find(time == 30, 1);
idx_60 = find(time == 60, 1);
idx_90 = find(time == 90, 1);
idx_120 = find(time == 120, 1);
ogtt_test_glucose = glucose_clean(:, [idx_fast, idx_30, idx_60, idx_90, idx_120]);
ogtt_test_insulin = insulin_clean(:, [idx_fast, idx_30, idx_60, idx_90, idx_120]);
G_fasting = glucose_noisy(:, idx_fast);   % [n_valid x 1]
I_fasting = insulin_noisy(:, idx_fast);
G_2h      = glucose_noisy(:, idx_120);    % [n_valid x 1]
% Apply ADA rules
[is_NGT, is_IGT, is_T2DM, err] = Classify_Diabetes_2H_OGTT(ogtt_test_glucose, ogtt_test_insulin);
% Calculate evaluation metrics
matsuda = Calculate_Matsuda_5_OGTT(ogtt_test_glucose, ogtt_test_insulin);
quicki = Calculate_QUICKI(G_fasting, I_fasting);

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
% Evaluation metrics figure
% ========================================================================
figure('Color', 'w');
hold on;
scatter(matsuda(is_NGT), quicki(is_NGT), 50, 'green', 'filled', 'DisplayName', 'Normal (NGT)');
scatter(matsuda(is_IGT), quicki(is_IGT), 50, 'blue', 'filled', 'DisplayName', 'Impaired (IGT)');
scatter(matsuda(is_T2DM), quicki(is_T2DM), 50, 'red', 'filled', 'DisplayName', 'Diabetes (T2DM)');
legend()
grid on;
box on;
xlabel('Matsuda Index', 'FontWeight', 'bold');
ylabel('QUICKI Index', 'FontWeight', 'bold');
title('Group Clusters: Matsuda vs QUICKI', 'FontSize', 12);
%% ========================================================================
% Summary figure
% =========================================================================
t_plot  = time;
n_show  = 100;   % trajectories per category in the plot

colors = struct('NGT',  [0.18 0.63 0.18], ...   % green
                'IGT',  [0.93 0.69 0.13], ...   % amber
                'T2DM', [0.80 0.15 0.15]);       % red

% Shared y-axis limits across all populations (99th percentile to avoid outliers)
all_G = [dataset_NGT.glucose_noisy; dataset_IGT.glucose_noisy; dataset_T2DM.glucose_noisy];
all_I = [dataset_NGT.insulin_noisy; dataset_IGT.insulin_noisy; dataset_T2DM.insulin_noisy];
ylim_G = [0, prctile(all_G(:), 99) * 1.1];
ylim_I = [0, prctile(all_I(:), 99) * 1.1];

figure('Name','Virtual Population - ADA Categories','Position',[80 80 1400 550]);

categories = {'NGT','IGT','T2DM'};
datasets   = {dataset_NGT, dataset_IGT, dataset_T2DM};

for col = 1:2   % col 1 = glucose, col 2 = insulin
    for r = 1:3
        ax = subplot(2, 3, (col-1)*3 + r);
        ds  = datasets{r};
        cat = categories{r};
        clr = colors.(cat);

        n_avail = ds.n;
        n_draw  = min(n_show, n_avail);
        idx     = randperm(n_avail, n_draw);
        if col == 1
            traces   = ds.glucose_noisy;
            med_val  = median(ds.glucose_noisy, 1);
            q1       = prctile(ds.glucose_noisy, 25, 1);
            q3       = prctile(ds.glucose_noisy, 75, 1);
            ylab     = 'Glucose (mmol/L)';
            ylim_cur = ylim_G;
        else
            traces   = ds.insulin_noisy;
            med_val  = median(ds.insulin_noisy, 1);
            q1       = prctile(ds.insulin_noisy, 25, 1);
            q3       = prctile(ds.insulin_noisy, 75, 1);
            ylab     = 'Insulin (mU/L)';
            ylim_cur = ylim_I;
        end
        
        % plot sampled traces
        plot(ax, t_plot, traces(idx,:)', 'Color', [clr 0.12], 'LineWidth', 0.5);
        xlabel('Time (min)'); ylabel(ylab);
        hold(ax, 'on');
        
        % plot IQR as shaded band
        fill_x = [t_plot, fliplr(t_plot)];
        fill_y = [q1, fliplr(q3)];
        h = fill(fill_x, fill_y, clr*0.8, 'EdgeColor', 'none');
        set(h, 'FaceAlpha', 0.6);
        % plot median on top
        plot(ax, t_plot, med_val, 'Color', clr*0.6, 'LineWidth', 2);
        
        xlabel(ax, 'Time (min)'); ylabel(ylab);
        title(ax, sprintf('%s  (n=%d)', cat, n_avail));
        xlim(ax, [0 300]);
        ylim(ax, ylim_cur);
        grid(ax,'on');
        hold(ax, 'off');

    end
end

sgtitle(sprintf('ADA classification | N_{total} = %d', n_valid));
