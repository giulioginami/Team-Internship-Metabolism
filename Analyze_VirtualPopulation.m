%% Analyze_VirtualPopulation.m
% Analyses the three ADA-labelled datasets (NGT, IGT, T2DM) from the EDES
% virtual population.
%
% Produces:
%   Figure 1 - Postprandial glucose:  median & mean per category
%   Figure 2 - Postprandial insulin:  median & mean per category
%   Figure 3 - Box plots of the 7 varied parameters per category
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc;

%% ========================================================================
% Load labelled virtual population
% =========================================================================
fprintf('Loading virtual_population_labelled.mat...\n');
load('virtual_population_labelled.mat', ...
     'dataset_NGT', 'dataset_IGT', 'dataset_T2DM');

datasets   = {dataset_NGT, dataset_IGT, dataset_T2DM};
cat_names  = {'NGT', 'IGT', 'T2DM'};
n_cats     = numel(cat_names);

time = dataset_NGT.time;   % common time vector

%% ========================================================================
% Colour scheme
% =========================================================================
% Each category: [line_color ; shade_color (for IQR fill)]
clr = struct( ...
    'NGT',  [0.18 0.63 0.18], ...   % green
    'IGT',  [0.93 0.69 0.13], ...   % amber
    'T2DM', [0.80 0.15 0.15]);      % red

line_styles = {'-', '--', ':'};   % NGT solid, IGT dashed, T2DM dotted

%% ========================================================================
% Pre-compute statistics for each category
% =========================================================================
for k = 1:n_cats
    ds = datasets{k};

    stats(k).n          = ds.n;                          %#ok<SAGROW>
    stats(k).G_median   = median(ds.glucose_noisy, 1);
    stats(k).G_mean     = mean(ds.glucose_noisy,   1);
    stats(k).G_q25      = quantile(ds.glucose_noisy, 0.25, 1);
    stats(k).G_q75      = quantile(ds.glucose_noisy, 0.75, 1);

    stats(k).I_median   = median(ds.insulin_noisy, 1);
    stats(k).I_mean     = mean(ds.insulin_noisy,   1);
    stats(k).I_q25      = quantile(ds.insulin_noisy, 0.25, 1);
    stats(k).I_q75      = quantile(ds.insulin_noisy, 0.75, 1);
end

%% ========================================================================
% Figure 1 — Postprandial Glucose
% =========================================================================
fig1 = figure('Name','Glucose Responses by ADA Category', ...
              'Position',[60 400 900 520]);

hold on;
h_leg = gobjects(n_cats * 2, 1);

for k = 1:n_cats
    c   = clr.(cat_names{k});
    ls  = line_styles{k};
    t   = time;

    % IQR shaded band
    fill([t, fliplr(t)], ...
         [stats(k).G_q25, fliplr(stats(k).G_q75)], ...
         c, 'FaceAlpha', 0.12, 'EdgeColor', 'none');

    % Median
    h_med = plot(t, stats(k).G_median, ...
                 'Color', c, 'LineWidth', 2.5, 'LineStyle', ls);

    % Mean  (slightly thinner, same colour, dashed variant)
    ls_mean = [ls '-'];   % e.g. '--' -> '---' not ideal; use fixed width instead
    h_mean = plot(t, stats(k).G_mean, ...
                  'Color', c*0.65, 'LineWidth', 1.5, 'LineStyle', ls);

    h_leg(2*k-1) = h_med;
    h_leg(2*k)   = h_mean;
end

% Reference lines
yline(5.6,  ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1, 'Label','5.6');
yline(7.0,  ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1, 'Label','7.0');
yline(11.1, ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1, 'Label','11.1');

xlabel('Time (min)', 'FontSize',12);
ylabel('Plasma Glucose (mmol/L)', 'FontSize',12);
title('Postprandial Glucose — Median & Mean by ADA Category', 'FontSize',13);
xlim([0 480]); grid on; box on;

legend_labels = {};
for k = 1:n_cats
    legend_labels{end+1} = sprintf('%s median  (n=%d)', cat_names{k}, stats(k).n); %#ok<SAGROW>
    legend_labels{end+1} = sprintf('%s mean',   cat_names{k});
end
legend(h_leg, legend_labels, 'Location','northeast', 'FontSize',9);

%% ========================================================================
% Figure 2 — Postprandial Insulin
% =========================================================================
fig2 = figure('Name','Insulin Responses by ADA Category', ...
              'Position',[60 60 900 520]);

hold on;
h_leg2 = gobjects(n_cats * 2, 1);

for k = 1:n_cats
    c   = clr.(cat_names{k});
    ls  = line_styles{k};
    t   = time;

    % IQR shaded band
    fill([t, fliplr(t)], ...
         [stats(k).I_q25, fliplr(stats(k).I_q75)], ...
         c, 'FaceAlpha', 0.12, 'EdgeColor', 'none');

    h_med2 = plot(t, stats(k).I_median, ...
                  'Color', c, 'LineWidth', 2.5, 'LineStyle', ls);
    h_mean2 = plot(t, stats(k).I_mean, ...
                   'Color', c*0.65, 'LineWidth', 1.5, 'LineStyle', ls);

    h_leg2(2*k-1) = h_med2;
    h_leg2(2*k)   = h_mean2;
end

xlabel('Time (min)', 'FontSize',12);
ylabel('Plasma Insulin (uIU/mL)', 'FontSize',12);
title('Postprandial Insulin — Median & Mean by ADA Category', 'FontSize',13);
xlim([0 480]); grid on; box on;

legend_labels2 = {};
for k = 1:n_cats
    legend_labels2{end+1} = sprintf('%s median  (n=%d)', cat_names{k}, stats(k).n); %#ok<SAGROW>
    legend_labels2{end+1} = sprintf('%s mean',   cat_names{k});
end
legend(h_leg2, legend_labels2, 'Location','northeast', 'FontSize',9);

%% ========================================================================
% Figure 3 — Box plots of varied parameters per category
% =========================================================================
param_names  = dataset_NGT.param_names;   % {'k1','k5','k6','k8','G_b','I_PL_b','BW'}
param_labels = {'k_1  (1/min)', 'k_5  (1/min)', 'k_6  (-)', ...
                'k_8  (-)',     'G_b  (mmol/L)', 'I_{PL,b}  (uIU/mL)', 'BW  (kg)'};
n_params = numel(param_names);

% Concatenate parameter matrices with group labels
all_params = [dataset_NGT.param_matrix; ...
              dataset_IGT.param_matrix; ...
              dataset_T2DM.param_matrix];

group_ids  = [ones(dataset_NGT.n,  1); ...
              2*ones(dataset_IGT.n, 1); ...
              3*ones(dataset_T2DM.n,1)];

box_colors = [clr.NGT; clr.IGT; clr.T2DM];   % 3 x 3

fig3 = figure('Name','Parameter Distributions by ADA Category', ...
              'Position',[980 60 1300 700]);

for p = 1:n_params
    subplot(2, 4, p);
    hold on;

    data_col = all_params(:, p);

    % Draw one boxplot group per category, coloured manually
    for k = 1:n_cats
        mask = (group_ids == k);
        vals = data_col(mask);

        % Box statistics
        q1   = quantile(vals, 0.25);
        q3   = quantile(vals, 0.75);
        med  = median(vals);
        mn   = mean(vals);
        iqr_ = q3 - q1;
        w_lo = max(min(vals), q1 - 1.5*iqr_);
        w_hi = min(max(vals), q3 + 1.5*iqr_);
        out_ = vals(vals < w_lo | vals > w_hi);

        c   = box_colors(k,:);
        bw  = 0.35;   % box half-width in x
        xc  = k;      % x-centre

        % Box body
        fill([xc-bw, xc+bw, xc+bw, xc-bw], [q1 q1 q3 q3], ...
             c, 'FaceAlpha',0.45, 'EdgeColor',c*0.6, 'LineWidth',1.5);
        % Median line
        plot([xc-bw, xc+bw], [med med], '-', 'Color',c*0.4, 'LineWidth',2.5);
        % Mean marker
        plot(xc, mn, 'd', 'Color',c*0.4, 'MarkerFaceColor',c*0.7, ...
             'MarkerSize',6, 'LineWidth',1);
        % Whiskers
        plot([xc xc], [w_lo q1], '-', 'Color',c*0.6, 'LineWidth',1.2);
        plot([xc xc], [q3 w_hi], '-', 'Color',c*0.6, 'LineWidth',1.2);
        % Whisker caps
        plot([xc-bw/2 xc+bw/2], [w_lo w_lo], '-', 'Color',c*0.6, 'LineWidth',1.2);
        plot([xc-bw/2 xc+bw/2], [w_hi w_hi], '-', 'Color',c*0.6, 'LineWidth',1.2);
        % Outliers
        if ~isempty(out_)
            plot(repmat(xc,size(out_)), out_, 'o', ...
                 'Color',c*0.6, 'MarkerSize',3, 'LineWidth',0.8);
        end
    end

    set(gca, 'XTick',1:n_cats, 'XTickLabel',cat_names, 'FontSize',10);
    ylabel(param_labels{p}, 'FontSize',10);
    title(param_labels{p}, 'FontSize',11);
    xlim([0.5, n_cats+0.5]); grid on; box on;
end

% Legend panel in the 8th (empty) subplot position
subplot(2,4,8); axis off;
for k = 1:n_cats
    fill(NaN, NaN, box_colors(k,:), 'FaceAlpha',0.45, ...
         'EdgeColor',box_colors(k,:)*0.6, 'LineWidth',1.5, ...
         'DisplayName', sprintf('%s  (n=%d)', cat_names{k}, stats(k).n));
    hold on;
end
legend('show', 'Location','center', 'FontSize',12);
title('Category legend', 'FontSize',11);

sgtitle('Parameter Distributions — NGT vs IGT vs T2DM', 'FontSize',14);

fprintf('\nDone. Three figures generated.\n');
