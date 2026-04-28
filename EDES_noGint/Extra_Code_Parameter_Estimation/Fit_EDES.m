function out = Fit_EDES(initial_par,input_data,individual,time,lb,ub)
%Fit EDES glucose-insulin model to  measured meal 
%challenge data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initial_par - initial guess for parameters to be optimised
%input_data  - structure of measured postprandial data
%              - must contain postprandial measurement of glucose, insulin, TG, NEFA
%              - must also contain body weight and meal composition
%individual  - specifies individual (row of input data array) to be fit.
%time        - vector of time for simulation
%lb          - lower bounds for parameter fitting (same size as initial_par)
%ub          - upper bounds for parameter fitting (same size as initial_par)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at s.d.odonovan@tue.nl
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%define options for lsqnonlin algorithm
lsq_options=optimset('Algorithm','trust-region-reflective','MaxFunEvals',1000,'TolX',1e-8,'Display','iter');
%Fit model to data
[p_opt,resnorm,residual,exitflag,output,lambda,jacobian]=lsqnonlin(@EDES_ErrorFunc,initial_par,lb,ub,lsq_options,input_data,individual,time);
 
out.p_opt    = p_opt;
out.resnorm  = resnorm;
out.residual = residual;
out.exitflag = exitflag;
out.output   = output;
out.lambda   = lambda;
out.jacobian = jacobian;