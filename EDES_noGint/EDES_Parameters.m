function out = EDES_Parameters(p_opt,input_data,individual)
%integrate optimised parameters and fixed parameters in one parameter
%vector for the EDES model
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p_opt           - vector of parameter values to be optimesed
% input_data      - struct of measured data needed to sepcify basal values
% individual      - specify individual (row in input data array) to be fitted
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at s.d.odonovan@tue.nl
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%glucose + insulin parameters (EDES Rozendaal 2018) (fixed parameters are
%fixed to values used in Rozendaal et al 2018)
out(1)  = p_opt(1); %0.0135; %k1 rate constant for glucose stomach emptying (fast)[1/min]
out(2)  = 0.28;     %k2 rate constant for glucose appearence from gut [1/min]
out(3)  = 6.07e-3;  %k3 rate constant for suppresstion of hepatic glucose release by change of plasma glucose
out(4)  = 2.35e-4;  %k4 rate constant for suppression of hepatic glucose release by delayed (remote) insulin
out(5)  = p_opt(2); %k5 rate constant for delayed insulin depedent uptake of glucose
out(6)  = p_opt(3); %k6 rate constant for stimulation of insulin production by the change of plasma glucose concentration (beta cell funtion)
out(7)  = 1.15;     %k7 rate constant for integral of glucose on insulin production (beta cell function)
out(8)  = 7.27;     %k8 rate constant for the simulation of insulin production by the rate of change in plasma glucose concentration (beta cell function)
out(9)  = 3.83e-2;  %k9 rate constant for outflow of insulin from plasma to interstitial space
out(10) = 2.84e-1;  %k10 rate constant for degredation of insulin in remote compartment
out(11) = 1.4;      %sigma shape factor (appearance of meal)
out(12) = 13.2;     %Km michaelis-menten coefficient for glucose uptake
out(13) = input_data.glucose(individual,1);%G_b basal plasma glucose [mmol/l]
out(14) = input_data.insulin(individual,1); %I_PL/_b basal plasma glucose [microU/ml]
out(15) = 0.043;    %basal hepatic glucose release