data = load('virtual_population_sparse.mat');

med = median(data.dataset_T2DM_sparse.param_matrix, 1); % median params of population

% ==========================================
% 1. DATA PREPARATION & INITIALIZATION
% ==========================================
t_data = [0, 30, 60, 90, 120, 150, 180, 210, 240, 360, 480]'; 
num_timepoints = length(t_data);
% Upload data

G_data_matrix = [data.dataset_T2DM_sparse.glucose_noisy];
I_data_matrix = [data.dataset_T2DM_sparse.insulin_noisy]; 
num_patients  = size(G_data_matrix, 1);                   

% Global Known Parameters
params.k7    = 1.15;
params.beta  = 1.0;
params.tau_i = 31.0;
params.tau_d = 3.0;   
params.k9    = 3.83e-2; 

% Pre-allocate a matrix to store the optimized parameters for each patient
% Rows = patients, Columns = [k6, k8]
k_optimized_all = zeros(num_patients, 2);

% Optimization setup
lb = [0, 0]; 
ub = [0.34, 10.0];
% Turning off the iterative display so it doesn't flood your command window for 200 loops
options = optimoptions('lsqnonlin', 'Display', 'off', 'StepTolerance', 1e-6);

% ==========================================
% 2. PATIENT LOOP
% ==========================================
disp('Starting individual optimizations for patients...');

for i = 1:num_patients
    % Extract individual patient data (column vectors)
    k_initial = data.dataset_T2DM_sparse.param_matrix(i, [3, 4]);  % [k6, k8] for patient i or median?
    G_pat = G_data_matrix(i, :);
    I_pat = I_data_matrix(i, :);
    
    % Patient-specific baselines
    params.Gb = G_pat(1); 
    params.Ib = I_pat(1); 
    
    % Pre-compute Derivative and Integral for THIS patient
    dG_pat = gradient(G_pat, t_data); 
    IntG_pat = cumtrapz(t_data, G_pat - params.Gb);
    
    % Define the cost function for THIS patient
    cost_function = @(k) compute_residuals(k, t_data, G_pat, dG_pat, IntG_pat, I_pat, params);
    
    % Run Optimization
    try
        k_opt = lsqnonlin(cost_function, k_initial, lb, ub, options);
        k_optimized_all(i, :) = k_opt;
    catch
        % In case a patient has messy data that causes the ODE solver to fail
        warning('Optimization failed for patient %d. Assigning NaNs.', i);
        k_optimized_all(i, :) = [NaN, NaN];
    end
end
clear k_opt
disp('Optimization complete!');

% Remove failed patients first
valid = ~any(isnan(k_optimized_all), 2);   % logical [n?1], true = converged
k_valid = k_optimized_all(valid, :);

% Representative values
k6_median = median(k_valid(:, 1));
k8_median  = median(k_valid(:, 2));

% Also useful: spread
k6_iqr = iqr(k_valid(:, 1));
k8_iqr = iqr(k_valid(:, 2));

fprintf('k6: median = %.4f  IQR = %.4f  (n=%d valid / %d total)\n', ...
        k6_median, k6_iqr, sum(valid), num_patients);
fprintf('k8: median = %.4f  IQR = %.4f\n', k8_median, k8_iqr);

figure;
subplot(1,2,1);
histogram(k_valid(:,1), 30); xlabel('k6'); ylabel('Count');
xline(median(k_valid(:,1)), 'r-', 'Median', 'LineWidth', 2);
xline(mean(k_valid(:,1)),   'b--','Mean',   'LineWidth', 2);
title('k6 distribution'); legend('show');

subplot(1,2,2);
histogram(k_valid(:,2), 30); xlabel('k8'); ylabel('Count');
xline(median(k_valid(:,2)), 'r-', 'Median', 'LineWidth', 2);
xline(mean(k_valid(:,2)),   'b--','Mean',   'LineWidth', 2);
title('k8 distribution'); legend('show');

% ==========================================
% 3. HELPER FUNCTIONS 
% ==========================================

function residuals = compute_residuals(k, t_data, G_data, dG_data, IntG_data, I_data, p)
    n     = length(t_data);
    I_sim = zeros(n, 1);
    I_sim(1) = I_data(1);   % start from fasting insulin

    for j = 1:n-1
        dt = t_data(j+1) - t_data(j);

        % dI/dt at current point j using known values (no interpolation)
        dIdt_j   = compute_dIdt(I_sim(j), G_data(j),   dG_data(j),   IntG_data(j),   k, p);

        % Euler predictor
        I_pred   = I_sim(j) + dt * dIdt_j;

        % dI/dt at next point j+1 using predicted I and known G values
        dIdt_jp1 = compute_dIdt(I_pred,   G_data(j+1), dG_data(j+1), IntG_data(j+1), k, p);

        % Trapezoidal corrector
        I_sim(j+1) = I_sim(j) + (dt/2) * (dIdt_j + dIdt_jp1);
    end

    residuals = I_sim - I_data(:);
end

function dIdt = compute_dIdt(I, G, dG, IntG, k, p)
    k6    = k(1); k8 = k(2);
    c3    = p.k7 * p.Gb / (p.beta * p.tau_i * p.Ib);
    i_pnc = (1/p.beta) * (k6*(G - p.Gb) + (p.k7/p.tau_i)*IntG + (p.k7/p.tau_i)*p.Gb + k8*p.tau_d*dG);
    i_liv = c3 * I;
    i_if  = p.k9 * (I - p.Ib);
    dIdt  = i_pnc - i_liv - i_if;
end