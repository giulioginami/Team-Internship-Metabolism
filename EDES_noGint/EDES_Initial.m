function [x0,c] = EDES_Initial(input_data,individual,parameters)
%Specify intitial values and model constants for EDES model from input data 
%and parameters
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%input_data - structure of measured meall challenge test data and
%             individidual information (BW, meal composition, ect).
%individual - sepcify individual (row of input_data) to be simulated.
%parameters - model parameters.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at s.d.odonova@tue.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define initial values for state variables
M_G_0     = 0; %intial mass of glucose in digestive tract (assume 0 as fasting)
G_PL_0    = input_data.glucose(individual,1); %fasting glucose concentration
I_PL_0    = input_data.insulin(individual,1);%fasting insulin concentration
I_d1_0    = 0; %insulin concentrtion in remote compartment


x0=[M_G_0,G_PL_0,I_PL_0,I_d1_0];

%% define model consntants

% conversion factor glucose - convert glucose from mg/l to mmol/l
c.f_G = 0.005551; %if mg/l

%conversion factor triglyceride - convert from mg/l to mmol/l
c.f_TG = 0.00113; 

%convert insulin from uIU/ml to mmol/l
c.f_I = 1; 

c.V_G = (260/sqrt(input_data.BW(individual)/70))/1000; %) volume of distribution for glucose (should be individualised somehow)
c.V_TG = (70/sqrt(input_data.BW(individual)/70))/1000; % 5/input_data.BW(individual) volume of distribution of triglycerides (volume of blood)
c.G_liv_b = parameters(15);%basal hepatic glucose production 
c.tau_i   = 31; %(min)
c.tau_d   = 3; %(min)
c.G_th_PL = 9; %threshold for renal extraction
c.c1      = 0.1;
%fixed to values specified in parameter vector (expected fasting values)
c.c2     = c.G_liv_b.*(parameters(12) + parameters(13))./parameters(13) - parameters(5).*c.f_I.*parameters(15);
%c.c3     = parameters(7).*parameters(13)./(c.f_I*c.tau_i.*parameters(14)).*c.t_integralwindow;