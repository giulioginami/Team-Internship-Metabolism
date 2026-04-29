% Load the dataset
population = 'NGT';  % choose: 'NGT', 'IGT', 'T2DM'

data   = load('virtual_population_labelled.mat');
dataset = data.(['dataset_' population]);

G_data_matrix = [dataset.glucose_noisy];
I_data_matrix = [dataset.insulin_noisy];
time_data     = [dataset.time];
param_matrix  = [dataset.param_matrix];

% Compute median trajectories across all patients
G_median = median(G_data_matrix, 1);
I_median = median(I_data_matrix, 1);

% Parameters
params.k7    = 1.15;
params.k9    = 3.83e-2;
params.tau_i = 31;
params.tau_d = 3;
params.beta  = 1;
params.c3    = (params.k7 * G_median(1)) / (params.beta * params.tau_i * I_median(1));

% Optimization setup
k_init = [median(param_matrix(:,3)), median(param_matrix(:,4))];
lb = [0, 0];
ub = [inf, inf];

options = optimoptions('lsqnonlin', 'Display', 'iter');

% Optimization
disp('Starting optimization...');
[k_opt, resnorm] = lsqnonlin(@(k) residuals(k, G_median, I_median, time_data, params), ...
                               k_init, lb, ub, options);

% Results
fprintf('\n--- Result of the population ---\n');
fprintf('k6 optimal: %.4f\n', k_opt(1));
fprintf('k8 optimal: %.4f\n', k_opt(2));
fprintf('Total quadratic error: %.4f\n', resnorm);

% Residual function
function res = residuals(k, G, I, t, params)
    k6 = k(1);
    k8 = k(2);

    num = [k8 * params.tau_d,  k6,  params.k7 / params.tau_i];
    den = [params.beta,  params.beta * (params.c3 + params.k9),  0];
    sys = tf(num, den);

    Gb = G(1);
    Ib = I(1);

    x_in   = G - Gb;
    y_real = I - Ib;

    y_sim = lsim(sys, x_in(:), t(:));

    res = y_real(:) - y_sim(:);
end
