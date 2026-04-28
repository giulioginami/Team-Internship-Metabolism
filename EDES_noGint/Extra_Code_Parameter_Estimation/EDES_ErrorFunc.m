function out = EDES_ErrorFunc(p_opt,input_data,individual,time)
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
% individual   - specifies individual (row of input data array) to be fit.
% time         - time span for model simulation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at s.d.odonovan@tue.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulate model for given parameter set

%form full parameter vector for simulation
parameters = EDES_Parameters(p_opt,input_data,individual);
%define intial values and model constants needed for simulation of eDES model
[initial_values,constants]=EDES_Initial(input_data,individual,parameters);

%specify options for ODE solver (Integrator function)
ODE_options = odeset('RelTol',1e-5);

%simulate model
[T,X] = ode45(@EDES_ODE,time,initial_values,ODE_options,parameters,constants,input_data,individual);

%% Calculate error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model fit error - data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%glucose error
measured_time_G=ismember(T,input_data.time_G);
G_err = (X(measured_time_G,2)' - input_data.glucose(individual,:))./max(input_data.glucose(individual,:));

%%insulin error
measured_time_I=ismember(T,input_data.time_I);
I_err = (X(measured_time_I,3)' - input_data.insulin(individual,:))./max(input_data.insulin(individual,:));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regularisation error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%error if AUC of rate of appearence of glucose from gut < meal content. 
k2  = parameters(2);
BW  = input_data.BW(individual);
f_G = constants.f_G;
V_G = constants.V_G;  

G_gut = k2.*(f_G/(V_G*BW)).*X(1:240,1);

AUC_G=trapz(G_gut);
AUC_G_norm = ((V_G*BW)/f_G)*AUC_G;

err_auc_G = abs((AUC_G_norm - input_data.meal.G(individual))./10000);



%error if triglyceride does not return to fasting value in 12 hours
G_5hours = input_data.glucose(individual,1)-X(300,2);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% total error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out=[max(input_data.glucose(individual,:)).*G_err,max(input_data.glucose(individual,:)).*I_err,err_auc_G,G_5hours];
