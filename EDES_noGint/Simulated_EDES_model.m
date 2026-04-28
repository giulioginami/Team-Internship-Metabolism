%Script to simulate the EDES Model (Rozendaal 2018)
%
%Originally written in MATLAB version 2019b,The MathWorks Inc., Natick,
%Massachusetts, United States.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for further information please contact Shauna O'Donovan at
%shauna.odonovan@wur.nl/s.d.odonovan@tue.nl
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% specify each parameter value ( can also be done using the
% EDES_Parameters.m function)
%glucose + insulin parameters Rozendaal et al. (2018)
parameters(1) = 0.0212;  %k1 rate constant for glucose stomach emptying (fast)[1/min]
parameters(2) = 0.28;      %k2 rate constant for glucose appearence from gut [1/min]
parameters(3) = 6.07e-3;   %k3 rate constant for suppresstion of hepatic glucose release by change of plasma glucose
parameters(4) = 2.35e-4;   %k4 rate constant for suppression of hepatic glucose release by delayed (remote) insulin
parameters(5) = 0.0174;  %k5 rate constant for delayed insulin depedent uptake of glucose
parameters(6) = 4.6237;  %k6 rate constant for stimulation of insulin production by the change of plasma glucose concentration (beta cell funtion)
parameters(7) = 1.15;      %k7 rate constant for integral of glucose on insulin production (beta cell function)
parameters(8) = 7.27;      %k8 rate constant for the simulation of insulin production by the rate of change in plasma glucose concentration (beta cell function)
parameters(9) = 3.83e-2;   %k9 rate constant for outflow of insulin from plasma to interstitial space
parameters(10) = 2.84e-1;  %k10 rate constant for degredation of insulin in remote compartment
parameters(11) = 1.4;      %sigma shape factor (appearance of meal)
parameters(12) = 13.2;     %Km michaelis-menten coefficient for glucose uptake
parameters(13) = 5;%G_b basal plasma glucose [mmol/l]
parameters(14) = 18; %I_PL/_b basal plasma glucose [microU/ml]
parameters(15) = 0.043;    %EGP_bbasal hepatic glucose release


%% Specify phenotypic parameters needed for simulation

sample_person.glucose = 5;    %fasting glucose (mmol/l)
sample_person.insulin = 18;   %fasting insulin (uIU/ml)
sample_person.BW      = 84.2; %body weight (kg)

%% specify meal composition
sample_person.meal.G  = 75000; %mass of glucose in meal (mg)

%% simulate model for given parameter set

%define intial values and model constants needed for simulation of M3al Model model
[initial_values,constants] = EDES_Initial(sample_person,1,parameters);

%specify options for ODE solver (Integrator function)
ODE_options = odeset('RelTol',1e-5);

%specify simulation time
time=0:1:480;

%simulate model
[T,X] = ode45(@EDES_ODE,time,initial_values,ODE_options,parameters,constants,sample_person,1);

%% plot fitting
%specify colour for plotting
plot_col = [0, 0.4470, 0.7410]; 

%open new figure pane
figure()

%plot glucose rate of apperance from gut
 subplot(1,3,1)
 k2 = parameters(2);
 BW = sample_person.BW;
 f_G = constants.f_G;
 V_G = constants.V_G; 
 
 G_gut = k2.*(f_G/(V_G*BW)).*X(:,1);
 
 AUC_G=trapz(G_gut);
 AUC_G = ((V_G*BW)/f_G)*AUC_G*0.001;
 
 plot(T,G_gut,'Color',plot_col,'LineWidth',1.5);
 hold on
 message=['AUC = ',num2str(AUC_G),'g'];
 text(240,max(G_gut),message,'HorizontalAlignment','right','Color',plot_col)
 xlabel('time (mins)')
 ylabel('glucose (mmol/l)');
 title('glucose from gut')

 %plot plasma glucose
 subplot(1,3,2)
 plot(T,X(:,2),'Color',plot_col,'LineWidth',1.5);
 hold on
 ylabel('glucose (mmol/l)');
 title('plasma Glucose')
 xlabel('time (mins)')
 
 %plot plasma insulin
 subplot(1,3,2)
 plot(T,X(:,4),'Color',plot_col,'LineWidth',1.5);
 hold on
 ylabel('insulin (uIU/ml)');
 title('plasma insulin')
 


