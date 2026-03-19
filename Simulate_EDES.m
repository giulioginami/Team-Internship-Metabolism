function out = Simulate_EDES(phenotypic_data,parameters,time,plot_col)
% Simulate EDES for a given parameter set
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% phenotypic_data - struct of information specifying fasting glucose,
%                   insulin, TG, and NEFA. Must also sepcify body weight (BW)
%                   and meal composition. 
% parameters      - parameter values to be simulated.
% time            - timespan for ode simulation
% plot_col        - Colour to plot line 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at
% shauna.odonovan@wur.nl/s.d.odonovan@tue.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulate model for given parameter set

%define intial values and model constants needed for simulation of M3al Model model
[initial_values,constants] = EDES_Initial(phenotypic_data,1,parameters);

%define global parameters for simulation
global t_saved G_PL_saved;
%initialise gloabl parameters
t_saved = 0;
G_PL_saved = phenotypic_data.glucose(1);

%specify options for ODE solver (Integrator function)
ODE_options = odeset('RelTol',1e-5,'OutputFcn',@integratorfunG);

%simulate model
[T,X] = ode45(@EDES_ODE,time,initial_values,ODE_options,parameters,constants,phenotypic_data,1);

%% Generate figure

 subplot(2,2,1)
 %glucose from gut
 k2 = parameters(2);
 BW = phenotypic_data.BW;
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


 subplot(2,2,2)
 %net hepatic glucose flux
 k3 = parameters(3);
 k4 = parameters(4);
 f_I = 1;
 G_liv_b = parameters(15);
 G_b = parameters(13);
 
 G_liv = G_liv_b - k4.*f_I.*X(:,5) - k3.*(X(:,2)-G_b);
 
 plot(T,G_liv,'Color',plot_col,'LineWidth',1.5);
 hold on
 xlabel('time (mins)')
 ylabel('glucose (mmol/l)');
 title('net hepatic glucose flux')
 
 
 
 
 subplot(2,2,3)
 %plot plasma glucose

 plot(T,X(:,2),'Color',plot_col,'LineWidth',1.5);
 hold on
 ylabel('glucose (mmol/l)');
 title('plasma Glucose')
 xlabel('time (mins)')
 
 
 subplot(2,2,4)
 %plasma insulin
 plot(T,X(:,4),'Color',plot_col,'LineWidth',1.5);
 hold on
 ylabel('insulin (uIU/ml)');
 title('plasma insulin')
 
 
 
 out=1;