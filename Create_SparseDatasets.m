%% Create_SparseDatasets.m
% Creates sparse versions of the three ADA-labelled datasets (NGT, IGT,
% T2DM) by retaining glucose and insulin values only at the clinically
% relevant time points:
%
%   t_sparse = [0, 30, 60, 90, 120, 150, 180, 210, 240, 360, 480] minutes
%
% Input  : virtual_population_labelled.mat
% Output : virtual_population_sparse.mat
%          Contains: dataset_NGT_sparse, dataset_IGT_sparse,
%                    dataset_T2DM_sparse  (plus the time vector t_sparse)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

%% ========================================================================
% Load labelled virtual population
% =========================================================================
fprintf('Loading virtual_population_labelled.mat...\n');
load('virtual_population_labelled.mat', ...
     'dataset_NGT', 'dataset_IGT', 'dataset_T2DM');

%% ========================================================================
% Define sparse time points and locate column indices
% =========================================================================
t_sparse    = [0, 30, 60, 90, 120, 150, 180, 210, 240, 360, 480];   % minutes
time_full   = dataset_NGT.time;               % full 0:1:480 vector

sparse_idx = zeros(1, numel(t_sparse));
for i = 1:numel(t_sparse)
    idx = find(time_full == t_sparse(i), 1);
    if isempty(idx)
        error('Time point t = %d min not found in the time vector.', t_sparse(i));
    end
    sparse_idx(i) = idx;
end

fprintf('Sparse time points : %s min\n', mat2str(t_sparse));
fprintf('Column indices     : %s\n\n',   mat2str(sparse_idx));

%% ========================================================================
% Helper: build a sparse sub-dataset from a labelled dataset struct
% =========================================================================
make_sparse = @(ds) struct( ...
    'category',       ds.category,                          ...
    'time',           t_sparse,                             ...  % [1 x 6]
    'glucose_noisy',  ds.glucose_noisy(:, sparse_idx),      ...  % [n x 6]
    'insulin_noisy',  ds.insulin_noisy(:, sparse_idx),      ...  % [n x 6]
    'glucose_clean',  ds.glucose_clean(:, sparse_idx),      ...  % [n x 6]
    'insulin_clean',  ds.insulin_clean(:, sparse_idx),      ...  % [n x 6]
    'param_matrix',   ds.param_matrix,                      ...  % [n x 7]
    'param_names',    {ds.param_names},                     ...
    'G_fasting',      ds.G_fasting,                         ...  % [n x 1]  t=0 noisy
    'G_2h',           ds.G_2h,                              ...  % [n x 1]  t=120 noisy
    'labels',         {ds.labels},                          ...
    'n',              ds.n                                  );

%% ========================================================================
% Build sparse datasets
% =========================================================================
dataset_NGT_sparse  = make_sparse(dataset_NGT);
dataset_IGT_sparse  = make_sparse(dataset_IGT);
dataset_T2DM_sparse = make_sparse(dataset_T2DM);

%% ========================================================================
% Print summary
% =========================================================================
fprintf('Sparse dataset summary (time points: %s min):\n', mat2str(t_sparse));
fprintf('  NGT  : %d patients  |  glucose/insulin: [%d x %d]\n', ...
    dataset_NGT_sparse.n,  dataset_NGT_sparse.n,  numel(t_sparse));
fprintf('  IGT  : %d patients  |  glucose/insulin: [%d x %d]\n', ...
    dataset_IGT_sparse.n,  dataset_IGT_sparse.n,  numel(t_sparse));
fprintf('  T2DM : %d patients  |  glucose/insulin: [%d x %d]\n', ...
    dataset_T2DM_sparse.n, dataset_T2DM_sparse.n, numel(t_sparse));

%% ========================================================================
% Save
% =========================================================================
save('virtual_population_sparse.mat', ...
     't_sparse', ...
     'dataset_NGT_sparse', 'dataset_IGT_sparse', 'dataset_T2DM_sparse', ...
     '-v7.3');

fprintf('\nSaved: virtual_population_sparse.mat\n');
fprintf('  Variables: t_sparse, dataset_NGT_sparse, dataset_IGT_sparse, dataset_T2DM_sparse\n');

%% ========================================================================
% Quick verification plot
% =========================================================================
categories   = {'NGT',  'IGT',  'T2DM'};
ds_list      = {dataset_NGT_sparse, dataset_IGT_sparse, dataset_T2DM_sparse};
clr_map      = {[0.18 0.63 0.18], [0.93 0.69 0.13], [0.80 0.15 0.15]};
n_show       = 80;

figure('Name','Sparse datasets : verification', 'Position',[80 80 1300 540]);

for col = 1:2
    for r = 1:3
        subplot(2, 3, (col-1)*3 + r);
        ds  = ds_list{r};
        c   = clr_map{r};

        n_draw = min(n_show, ds.n);
        idx    = randperm(ds.n, n_draw);

        if col == 1
            vals = ds.glucose_noisy;
            ylab = 'Glucose (mmol/L)';
        else
            vals = ds.insulin_noisy;
            ylab = 'Insulin (mU/L)';
        end

        % Individual sparse traces (markers only)
        plot(repmat(t_sparse, n_draw, 1)', vals(idx,:)', ...
             'o-', 'Color', [c 0.15], 'MarkerSize', 3, 'LineWidth', 0.6);
        hold on;

        % Median across all patients
        plot(t_sparse, median(vals, 1), 's-', ...
             'Color', c*0.55, 'LineWidth', 2.2, 'MarkerSize', 7, ...
             'MarkerFaceColor', c*0.55, 'DisplayName', 'Median');

        xlabel('Time (min)'); ylabel(ylab);
        title(sprintf('%s : %s  (n=%d)', categories{r}, ylab, ds.n));
        set(gca, 'XTick', t_sparse);
        xlim([-20, 500]); grid on; box on;
    end
end

sgtitle('Sparse datasets: t = [0, 30, 60, 90, 120, 150, 180, 210, 240, 360, 480] min', 'FontSize',13);
