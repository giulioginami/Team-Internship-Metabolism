function out = EDES_ErrorFunc(p_opt,input_data,row,time)
%Error function between EDES model simulation for a given
%parameter set and supplied measured meal challenge test data
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p_opt        - vector of parameter values for which error is being
%                calculated.
% input_data   - struct of measured challenge test data for caculating error
%              - mean and standard deviation values are required for
%                glucose and insulin (need to extend to TG and NEFA)
%              - vector of sampling time points are also required.
%              - any additional variables required for model simulation.
% row          - specifies individual (row of input data array) to be fit.
% time         - time span for model simulation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at
% shauna.odonovan@wur.nl/s.d.odonovan@tue.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulate model for given parameter set

%form full parameter vector for simulation
parameters = EDES_Parameters(p_opt,input_data,row);
%define intial values and model constants needed for simulation of eDES model
[initial_values,constants]=EDES_Initial(input_data,row,parameters);

%define global parameters for simulation
global t_saved G_PL_saved;
%initialise gloabl parameters
t_saved = 0;
G_PL_saved = input_data.glucose(row,1);

%specify options for ODE solver (Integrator function)
ODE_options = odeset('RelTol',1e-5,'OutputFcn',@integratorfunG);

%simulate model
[T,X] = ode45(@EDES_ODE,time,initial_values,ODE_options,parameters,constants,input_data,row);

%% Calculate error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model fit error - data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%glucose error
measured_time_G=ismember(T,input_data.time_G);
G_err = (X(measured_time_G,2)' - input_data.glucose(row,:))./max(input_data.glucose(row,:));

%%insulin error
measured_time_I=ismember(T,input_data.time_I);
I_err = (X(measured_time_I,4)' - input_data.insulin(row,:))./max(input_data.insulin(row,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regularisation error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%error if AUC of rate of appearence of glucose from gut < meal content. 
k2  = parameters(2);
BW  = input_data.BW(row);
f_G = constants.f_G;
V_G = constants.V_G;  

G_gut = k2.*(f_G/(V_G*BW)).*X(1:240,1);

AUC_G=trapz(G_gut);
AUC_G_norm = ((V_G*BW)/f_G)*AUC_G;

err_auc_G = abs((AUC_G_norm - input_data.meal.G(row))./10000);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% total error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out=[max(input_data.glucose(row,:)).*G_err,max(input_data.glucose(row,:)).*I_err,err_auc_G];

