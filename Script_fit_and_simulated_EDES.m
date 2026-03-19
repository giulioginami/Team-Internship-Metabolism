%Script for estimating parameters for the EDES from measured meal
%challenge test data. 
%
%Originally written in MATLAB version 2019b,The MathWorks Inc., Natick,
%Massachusetts, United States.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for further information please contact Shauna O'Donovan at
%shauna.odonovan@wur.nl/s.d.odonovan@tue.nl
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%the data should have the following structure.
%input_data.glucose    - measured time series of plasma glucose during meal(mmol/l).
%input_data.insulin    - measured time series of plasma insulin during meal (uIU/ml).
%input_data.TG         - measured time series of plasma triglyceride during meal(mmol/l).
%input_data.NEFA       - measured time series of plasma NEFA during meal (mmol/l).
%input_data.BW         - body weight (kg).
%input_data.meal.G     - mass of glucose in the meal (mg).
%input_data.meal.TG    - mass of lipid/triglyceride in the meal (mg).
%input_data.time_G     - time points/sampling schedule for glucose measurements (mins).
%input_data.time_I     - time points/sampling schedule for insulin measurements (mins).
%input_data.time_TG    - time points/sampling schedule for triglyceride measurements (mins).
%input_data.time_NEFA  - time points/sampling schedule for NEFA measurements (mins).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('sample_data.mat')
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fit model to measured data using lsqnonlin;
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_par_sets = 5; %specify how many initial parameter sets are used.
row = 1; %specify which row of input_data arrays will be fit.

fitting = Fit_EDES_LatinHyperCube(num_par_sets,sample_data,row);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%To visualise the model fits - Multiple Fits

time = 0:1:240; % 

Plot_MultiFit_EDES(fitting,sample_data,1,time);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%To visualise the model simulation for a specified parameter set.

%% specify an individual optimatl parameter set (from fitting)
p_opt = [0.0104,0.0422,2.4134];

%% plot fitting
plot_colour = [0, 0.4470, 0.7410]; %specify colour for plotting

figure()

Plot_EDES(p_opt,sample_data,time,plot_colour);

%% alternatively specify each parameter value in turn
%glucose + insulin parameters Rozendaal et al. (2018)
parameters(1) = p_opt(1);  %k1 rate constant for glucose stomach emptying (fast)[1/min]
parameters(2) = 0.28;      %k2 rate constant for glucose appearence from gut [1/min]
parameters(3) = 6.07e-3;   %k3 rate constant for suppresstion of hepatic glucose release by change of plasma glucose
parameters(4) = 2.35e-4;   %k4 rate constant for suppression of hepatic glucose release by delayed (remote) insulin
parameters(5) = p_opt(2);  %k5 rate constant for delayed insulin depedent uptake of glucose
parameters(6) = p_opt(3);  %k6 rate constant for stimulation of insulin production by the change of plasma glucose concentration (beta cell funtion)
parameters(7) = 1.15;      %k7 rate constant for integral of glucose on insulin production (beta cell function)
parameters(8) = 7.27;      %k8 rate constant for the simulation of insulin production by the rate of change in plasma glucose concentration (beta cell function)
parameters(9) = 3.83e-2;   %k9 rate constant for outflow of insulin from plasma to interstitial space
parameters(10) = 2.84e-1;  %k10 rate constant for degredation of insulin in remote compartment
parameters(11) = 1.4;      %sigma shape factor (appearance of meal)
parameters(12) = 13.2;     %Km michaelis-menten coefficient for glucose uptake
parameters(13) = sample_data.glucose(row,1);%G_b basal plasma glucose [mmol/l]
parameters(14) = sample_data.insulin(row,1); %I_PL/_b basal plasma glucose [microU/ml]
parameters(15) = 0.043;    %EGP_bbasal hepatic glucose release


%% Specify phenotypic parameters for simulation

sample_person.glucose = 5;    %fasting glucose (mmol/l)
sample_person.insulin = 18;   %fasting insulin (uIU/ml)
sample_person.BW      = 84.2; %body weight (kg)

%% specify meal composition
sample_person.meal.G  = 75000; %mass of glucose in meal (mg)


%% plot fitting
plot_colour = [0, 0.4470, 0.7410]; %specify colour for plotting

figure()

Simulate_EDES(sample_person,parameters,time,plot_colour);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%