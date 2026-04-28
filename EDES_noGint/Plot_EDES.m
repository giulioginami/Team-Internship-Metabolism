function out = Plot_EDES(input_data,individual,p_opt,time,plot_col)
% Plot measured data and EDES Model simulation for a given parameter set
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input_data - struct of measured challenge test data must contain
%              vectors of time series of mean values and standard deviations
%              for glucose, insulin with that nomenclature.
%              Must also contain vector of time points corresponding to
%              sampling time points of measured data. 
%individual  - individual being fitted - index for the row of the input
%              array to be used for fitting.
% p_opt      - parameter values to be simulated.
% time       - timespan for ode simulation
% plot_col   - Colour to plot line 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at s.d.odonovan@tue.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% simulate model for given parameter set
%generate complete set of model parameters from p_opt vector
parameters = EDES_Parameters(p_opt,input_data,individual);

%define intial values and model constants needed for simulation of eDES model
[initial_values,constants] = EDES_Initial(input_data,individual,parameters);

%specify options for ODE solver (Integrator function)
ODE_options = odeset('RelTol',1e-5);

%simulate model
[T,X] = ode45(@EDES_ODE,time,initial_values,ODE_options,parameters,constants,input_data,individual);

%% Generate figure

 subplot(2,2,1)
 %glucose from gut
 k2 = parameters(2);
 BW = input_data.BW(individual);
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
 xticks(input_data.time_TG)
 

 subplot(2,2,2)
 %net hepatic glucose flux
 k3 = parameters(3);
 k4 = parameters(4);
 f_I = 1;
 G_liv_b = parameters(15);
 G_b = parameters(13);%input_data.glucose(individual,1);
 
 G_liv = G_liv_b - k4.*f_I.*X(:,4) - k3.*(X(:,2)-G_b);
 
 plot(T,G_liv,'Color',plot_col,'LineWidth',1.5);
 hold on
 xlabel('time (mins)')
 ylabel('glucose (mmol/l)');
 title('net hepatic glucose flux')
 xticks(input_data.time_TG)
 
 
 
 
 subplot(2,2,3)
 %plot plasma glucose

 plot(T,X(:,2),'Color',plot_col,'LineWidth',1.5);
 hold on
 plot(input_data.time_G,input_data.glucose(individual,:),'kx','MarkerSize',10,'LineWidth',1.5)
 ylabel('glucose (mmol/l)');
 title('plasma Glucose')
 xlabel('time (mins)')
 xticks(input_data.time_TG)
 
 
 subplot(2,2,4)
 %plasma insulin
 plot(T,X(:,3),'Color',plot_col,'LineWidth',1.5);
 hold on
 plot(input_data.time_I,input_data.insulin(individual,:),'kx','MarkerSize',10,'LineWidth',1.5)
 %plot(T,zeros(size(T)),'k--');
 ylabel('insulin (uIU/ml)');
 title('plasma insulin')
 xticks(input_data.time_TG)

 
 out=1;