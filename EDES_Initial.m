function [x0,c] = EDES_Initial(input_data,row,parameters)
%Specify intitial values and model constants for Mixed Meal Model (M3al
%Model)from input data and parameters
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%input_data - structure of measured meall challenge test data and
%             individidual information (BW, meal composition, ect).
%row        - sepcify row of input_data to be simulated.
%parameters - model parameters.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at
% shauna.odonovan@wur.nl/s.d.odonova@tue.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define initial values for state variables
M_G_0     = 0; %intial mass of glucose in digestive tract (assume 0 as fasting)
G_PL_0    = input_data.glucose(row,1); %fasting glucose concentration
G_int_0   = 0; %integrated plasma glucose (assume 0 as fasting)
I_PL_0    = input_data.insulin(row,1);%fasting insulin concentration
I_d1_0    = 0; %insulin concentrtion in remote compartment


x0=[M_G_0,G_PL_0,G_int_0,I_PL_0,I_d1_0];

%% define model consntants

% conversion factor glucose - convert glucose from mg/l to mmol/l
c.f_G = 0.005551; %if mg/l

%conversion factor triglyceride - convert from mg/l to mmol/l


%convert insulin from uIU/ml to mmol/l
c.f_I = 1; 

c.V_G     = 17/70; %volume of distribution for glucose 

c.G_liv_b = parameters(15);%basal hepatic glucose concentration 
c.tau_i   = 31; %(min)
c.tau_d   = 3; %(min)
c.G_th_PL = 9; %threshold for renal extraction
c.t_integralwindow = 30; %30
c.c1      = 0.1;
%fixed to healthy values (from Rozendaal et al. 2018)
%c.c2    = c.G_liv_b.*(parameters(13) + 4.9)/4.9 - parameters(5)*c.f_I*6.8;
%c.c3    = parameters(7).*4.9/(c.f_I*c.tau_i.*6.8).*c.t_integralwindow;
%fixed to values specified in parameter vector (expected fasting values)
c.c2     = c.G_liv_b.*(parameters(12) + parameters(13))./parameters(13) - parameters(5).*c.f_I.*parameters(15);
c.c3     = parameters(7).*parameters(13)./(c.f_I*c.tau_i.*parameters(14)).*c.t_integralwindow;

