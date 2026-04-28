%Script for Estimating parameters for EDES Model 
%
%Originally written in MATLAB version 2021a,The MathWorks Inc., Natick,
%Massachusetts, United States.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%for further information please contact Shauna O'Donovan at s.d.odonovan@tue.nl
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%the data should have the following structure.
%input_data.glucose    - measured time series of plasma glucose during meal(mmol/l).
%input_data.insulin    - measured time series of plasma insulin during meal (uIU/ml).
%input_data.BW         - body weight (kg).
%input_data.meal.G     - mass of glucose in the meal (mg).
%input_data.time_G     - time points/sampling schedule for glucose measurements (mins).
%input_data.time_I     - time points/sampling schedule for insulin measurements (mins).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('sample_data.mat')
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%fit model to measured data for individual 1 using lsqnonlin;
%
%-multiple initial values defined by LatinHyperCube sampling of user 
% specified ranges for each parameter.
%-the number of parameters being estimated (line29) and the ranges for each
% parameter (line32/33)can be adjusted within the Fit_M3al_Model_LatinHyperCube
% function itself.
%-this function uses the cost function as defined in M3al_Model_ErrorFunc.
%-the parameters being estimated and those that are fixed are specified in
% the function M3al_Model_Parameters.
%-model constants and conversion factors are specified in the function
% M3al_Model_Initial. 
%-the equations underlying the M3al Model are implemented in the
% M3al_Model_ODE function. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%specify which individual from your sample data you are fitting
individual = 1;

%specify inital parameter guess
initial_par = [0.002,0.03,2];

%specify upper and lower bound for the optimisation algorithm
lb = [0,0,0];
ub = [0.5,1,10];

%specify for how long the model should simulate to best calculate the error
%with your data
time_span=0:1:480;


fitting = Fit_EDES(initial_par,sample_data,individual,time_span,lb,ub);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%To visualise the model fit
%specify colour for plotting
plot_colour = [0, 0.4470, 0.7410]; 
figure()

Plot_EDES(sample_data,individual,fitting.p_opt,time_span,plot_colour);

