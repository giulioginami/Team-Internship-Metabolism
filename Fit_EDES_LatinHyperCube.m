function out = Fit_EDES_LatinHyperCube(num_par_sets,input_data,row)
%Fit EDES model to measured meal challenged
%test data from Intervention Study X using multiple initial values for lsqnonlin.
%
%Initial values are generated using a LatinHyperCube sampling approach across
%parameter regions specified in the function below.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%num_par_sets    - number of initial parameter sets to be tested
%input_data      - input data for fitting - structure containing arrays of
%                  measured time series of glucose, insulin,TG, and NEFA as
%                  well as body weight and meal composition.
%row             - individual being fitted - index for the row of the input
%                  array to be used for fitting.
%meal_parameters - vector of pre-optimised parameters for glucose-inuslin
%                  parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%out - array of 'optimal' parameter values
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at
% shauna.odonovan@wur.nl/s.d.odonovan@tue.nl
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%specify simulation time (note this may be longer than measured time due to
%meet the regularisations terms; currently 720 minutes is adivsed)
time = 0:1:480;

%specify number of parameters being optimised
num_par = 3;

%specify upper and lower bound for each parameter
lb = [0.005,0,0];
ub = [0.1,1,15];
d  = ub - lb;

% Latin Hypercube Sampling of parameter space
lhs = lhsdesign(num_par_sets, num_par);   
% Adjust parameter sets for MC to reference values:
initial_pars = lhs.*d + lb; %scale parameters

%specify output structure
out.p_opt   = zeros(num_par_sets,num_par);
out.stop    = zeros(num_par_sets,1);
out.initial_par = initial_pars;
out.resnorm = zeros(num_par_sets,1);


for i=1:num_par_sets
    %try output=Fit_EDES(initial_pars(i,:),input_data,row,time,lb,ub);
        output=Fit_EDES(initial_pars(i,:),input_data,row,time,lb,ub);
        out.p_opt(i,:)=output.p_opt;
        out.stop(i)=output.exitflag;
        out.resnorm(i)=output.resnorm;
    %catch err 
    %        disp('Optimization has failed. Resampling.');
    %        out.p_opt(i,:)=NaN.*ones(1,3);
    %        out.stop(i)=9;
    %        out.resnorm(i)=NaN;
    %        %rethrow(err)
    %end
end