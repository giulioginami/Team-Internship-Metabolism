%% Compare_Japan_vs_Virtual.m
% Loads the labelled virtual population and the converted Japan dataset,
% and generates a comparison plot of their glucose and insulin trajectories 
% across the three ADA categories (NGT, IGT, T2DM).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

%% ========================================================================
% Load Datasets
% =========================================================================
fprintf('Loading virtual_population_labelled.mat...\n');
% Ensure you have run Generate_VirtualPopulation.m and Label_VirtualPopulation.m
load('virtual_population_labelled.mat', 'dataset_NGT', 'dataset_IGT', 'dataset_T2DM');

fprintf('Loading japan_population_labelled.mat...\n');
% Ensure you have run Convert_Japan_Dataset.m
load('japan_population_labelled.mat', 'japan_population');

%% ========================================================================
% Prepare Data Subsets
% =========================================================================
time_j = japan_population.time; % OGTT time stamps: [0, 30, 60, 90, 120]

% Extract Japan glucose data by category
ds_japan_G = {japan_population.glucose_noisy(japan_population.is_NGT, :), ...
              japan_population.glucose_noisy(japan_population.is_IGT, :), ...
              japan_population.glucose_noisy(japan_population.is_T2DM, :)};

% Extract Japan insulin data by category
ds_japan_I = {japan_population.insulin_noisy(japan_population.is_NGT, :), ...
              japan_population.insulin_noisy(japan_population.is_IGT, :), ...
              japan_population.insulin_noisy(japan_population.is_T2DM, :)};

categories = {'NGT', 'IGT', 'T2DM'};
ds_virtual = {dataset_NGT, dataset_IGT, dataset_T2DM};

colors = struct('NGT',  [0.18 0.63 0.18], ...   % green
                'IGT',  [0.93 0.69 0.13], ...   % amber
                'T2DM', [0.80 0.15 0.15]);      % red

%% ========================================================================
% Plot Comparison
% =========================================================================
figure('Name','Comparison: Virtual vs Japan Population','Position',[100 100 1400 800], 'Color', 'w');

for col = 1:2 % Column 1 = Glucose, Column 2 = Insulin
    for r = 1:3 % Row 1 = NGT, Row 2 = IGT, Row 3 = T2DM
        ax = subplot(2, 3, (col-1)*3 + r);
        cat = categories{r};
        clr = colors.(cat);
        hold(ax, 'on');

        % --- 1. Extract and Format Data ---
        ds_v = ds_virtual{r};
        time_v = ds_v.time;
        
        if col == 1
            data_v = ds_v.glucose_noisy;
            data_j = ds_japan_G{r};
            ylab = 'Glucose (mmol/L)';
        else
            data_v = ds_v.insulin_noisy;
            data_j = ds_japan_I{r};
            ylab = 'Insulin (mU/L)';
        end

        % --- 2. Plot Virtual Population (Continuous Shaded IQR) ---
        med_v = median(data_v, 1);
        q1_v  = prctile(data_v, 25, 1);
        q3_v  = prctile(data_v, 75, 1);

        % Fill IQR area
        fill_x = [time_v, fliplr(time_v)];
        fill_y = [q1_v, fliplr(q3_v)];
        h_v = fill(fill_x, fill_y, clr, 'EdgeColor', 'none', 'FaceAlpha', 0.25);
        
        % Plot median line
        p_v = plot(ax, time_v, med_v, '-', 'Color', clr, 'LineWidth', 2.5);

        % --- 3. Plot Real Japan Population (Discrete Markers + Error bars) ---
        med_j = median(data_j, 1);
        q1_j  = prctile(data_j, 25, 1);
        q3_j  = prctile(data_j, 75, 1);

        % Plot discrete points with IQR error bars
        p_j = errorbar(ax, time_j, med_j, med_j - q1_j, q3_j - med_j, ...
            'o--', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', 'k', 'LineWidth', 1.5, 'CapSize', 8);

        % --- 4. Chart Formatting ---
        xlabel(ax, 'Time (min)', 'FontWeight', 'bold');
        ylabel(ax, ylab, 'FontWeight', 'bold');
        title(ax, sprintf('%s (Virtual n=%d, Japan n=%d)', cat, ds_v.n, size(data_j,1)));
        
        % Constrain X-axis to 150 minutes since the Japan dataset stops at 120 minutes
        xlim(ax, [-5 140]); 
        grid(ax, 'on');
        set(ax, 'Layer', 'top');

        % Add a single legend to the first plot
        if r == 1 && col == 1
            legend([p_v, h_v, p_j], {'Virtual (Median)', 'Virtual (IQR)', 'Japan (Median \pm IQR)'}, ...
                'Location', 'northwest', 'FontSize', 10);
        end
    end
end

sgtitle('Comparison: Virtual Population vs. Japan Clinical Dataset', 'FontSize', 16, 'FontWeight', 'bold');