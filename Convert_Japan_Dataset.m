file_location = 'pone.0143880.s010.xls';
T_raw = readtable(file_location, 'VariableNamingRule', 'preserve');
T_raw.Properties.VariableNames = regexprep(T_raw.Properties.VariableNames, '\s+', ' ');

rawNames = { 'No', 'sex (male1)', 'age', 'BW', 'BMI', 'type',  ...
    'GIR', 'incremental AUC IRI(10)', 'incremental AUC IRI(10-90)', 'oral DI', 'Matuda index', 'QUICKI', ...
    'O-PG(0)', 'O-PG(30)', 'O-PG(60)',	'O-PG(90)',	'O-PG(120)', 'O- IRI(0)', 'O-IRI(30)', 'O-IRI(60)',	'O-IRI(90)', 'O-IRI(120)'
};
structNames = { 'id', 'sex', 'age', 'BW', 'BMI', 'labels',  ...
    'GIR', 'incr_AUC_IRI_10', 'incr_AUC_IRI_10_90', 'DI', 'matsuda', 'quicki', ...
    'PG_0_mgdl', 'PG_30_mgdl', 'PG_60_mgdl',	'PG_90_mgdl',	'PG_120_mgdl', 'IR_0', 'IR_30', 'IR_60',	'IR_90', 'IR_120'
};

T_roi = T_raw(:, rawNames);
T_roi = renamevars(T_roi, rawNames, structNames);
% remove nan entries
T_roi = T_roi(~any(isnan(T_roi{:, 13:end-1}), 2), :);

S = table2struct(T_roi, 'ToScalar', true);
% format the time stamps as 5-OGTT
S.time = [0.0, 30.0, 60.0, 90.0, 120.0];
% format glucose and insulin data
S.glucose_noisy = [S.PG_0_mgdl, S.PG_30_mgdl, S.PG_60_mgdl, S.PG_90_mgdl, S.PG_120_mgdl];
% convert glucose mg/dL to mmol/L
S.glucose_noisy = S.glucose_noisy ./ 18.0;
S.insulin_noisy = [S.IR_0, S.IR_30, S.IR_60, S.IR_90, S.IR_120];
S.G_fasting = S.glucose_noisy(1);
S.G_2h = S.glucose_noisy(5);

S.param_names = {'k1','k5', 'k6', 'k8', 'G_b', 'I_PL_b', 'BW'};
S.param_matrix = [S.DI, zeros(length(S.BW)), zeros(length(S.BW), 1), zeros(length(S.BW), 1), S.glucose_noisy(:, 1), S.insulin_noisy(:, 1), S.BW];
S.N_attempted = S.id(end);
S.n_valid = length(S.id);
S.labels = string(S.labels);
S.is_NGT = (S.labels == 'NGT');
S.is_IGT = (S.labels == 'IGT');
S.is_T2DM = (S.labels == 'T2DM');

japan_population = S;
save('japan_population_labelled.mat', 'japan_population');

