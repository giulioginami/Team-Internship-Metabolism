function dxdt = EDES_ODE(t,variables,parameters,constants,input_data,row)
%% Implementation of Eindhoven Diabetes Education Simulator (Rozendaal et al. 
%  (2018) model of postprandial glucose and insulin dynamics
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%t          - time value
%variables  - vector of state variables at given time point time 
%             at t=0 variables = initial values
%parameters - vector of parameter values
%constants  - struct of system constants
%input_data - necessary input_data, meal composition, body mass ect.
%row        - row of input data array being simulated
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for further information contact Shauna O'Donovan at
% shauna.odonovan@wur.nl/s.d.odonova@tue.nl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% define state variables
M_G_gut    = variables(1); %glucose mass in gut (mg)
G_PL       = variables(2); %plasma glucose concentration (mmol/l)
G_int      = variables(3); %integral of plasma glucose (PID controler for insulin secretion)
I_PL       = variables(4); %plasma insulin concentration (mU/l)
I_d1       = variables(5); %delayed/remote insulin concentration 

%% model constants
c = constants;
f_G     = c.f_G;     %conversion factor glucose - convert mg/l to mmol/l
f_I     = c.f_I;     %conversion factor insulin - convert uIU/ml to mmol/l
V_G     = c.V_G;     %glucose distribution volume (L/kg)
tau_i   = c.tau_i;   % intergration time constant (min)
tau_d   = c.tau_d;   %differential time constant (min)
G_th_PL = c.G_th_PL; % threshold for renal glucose extraction (mmol/L)
c1      = c.c1;      %constant term in renal extraction (l/min) (rate constant for glomerular filtration)
t_integralwindow = c.t_integralwindow; %Lower bound of moving time window of G_int

%% model input
D_meal_G  = input_data.meal.G(row);  %total amount of carbohydrates ingensted (mg)
BW        = input_data.BW(row);      %body weight (kg)

%% model parameters

%glucose + insulin parameters (EDES Rozendaal et al. 2018)
k1     = parameters(1);  % rate constant for glucose stomach emptying (fast)[1/min]
k2     = parameters(2);  % rate constant for glucose appearance from gut [1/min]
k3     = parameters(3);  % rate constant for suppression of hepatic glucose release by change of plasma glucose [1/min] 
k4     = parameters(4);  % rate constant for suppression of hepatic glucose release by  delayed insulin (remote compartment) [1/min]
k5     = parameters(5);  % rate constant for delayed insulin depedent uptake of glucose[1/min]
k6     = parameters(6);  % rate constant for stimulation of insulin production by the change of plasma glucose concentration[1/min] (proportional)
k7     = parameters(7);  % rate constant for integral of glucose on insulin production[1/min] (integral)
k8     = parameters(8);  % rate constant for the simulation of insulin production by the rate of change in plasma glucose concentration [1/min] (derivative)
k9     = parameters(9);  % rate constant for outflow of insulin from plasma to remote compartment[1/min] 
k10    = parameters(10); % rate constant for utilisation of insulin in remote compartment (degredation)
sigma  = parameters(11); % shape factor (appearance of meal)[-]
KM     = parameters(12); % michaelis-menten coefficient for glucose uptake[mmol/l]
G_b    = parameters(13); % basal plasma glucose [mmol/l]
I_PL_b = parameters(14); % basal plasma glucose [microU/ml]
G_liv_b = parameters(15);%basal hepatic glucose release

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Model equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Appearance of glucose from meal

% appearance fo glucose from meal as a function of time
G_meal = sigma*(k1.^sigma)*t.^(sigma-1) .* exp(-(k1.*(t)).^sigma).* D_meal_G;
%glucose mass in gut
D.M_G_gut = G_meal - k2*M_G_gut;

%% plasma glucose
%net glucose flux across liver
G_liv = G_liv_b - k4.*f_I.*I_d1 - k3.*(G_PL-G_b);
%glucose concentration in gut
G_gut = k2.*(f_G/(V_G*BW)).*M_G_gut;
%insulin independent glucose utilisation (brain, erythrocytes ect.)
U_ii = G_liv_b*((KM + G_b)./G_b).*(G_PL./(KM + G_PL));
%insulin dependent glucose utilisation (liver,muscle,adipose)
U_id = k5.*f_I.*I_d1.*(G_PL./(KM + G_PL));
%renal extraction of plasma glucose
U_ren = (c1./(V_G*BW).*(G_PL - G_th_PL))*(G_PL > G_th_PL);

%rate of change of plasma glucose
D.G_PL = G_liv + G_gut - U_ii - U_id - U_ren;

%% plasma insulin
global t_saved G_PL_saved

t_lowerbound = t - t_integralwindow;
if (t > t_integralwindow) && (length(t_saved)>1) && (length(t_saved) == length(G_PL_saved))
    G_PL_lowerbound = interp1(t_saved,G_PL_saved,t_lowerbound, 'spline');
else
    G_PL_lowerbound = G_PL_saved(1);  % is called when t < t_integralwindow, or if there is no saved step yet (steps are only saved at pre-defined time points)
end
%intagration og G_pl-G_b over the interval t_int to t.
D.G_int = (G_PL -G_b) - (G_PL_lowerbound - G_b);
%production/release of insulin by pancreas
I_pnc = (f_I.^-1).*(k6.*(G_PL - G_b) + (k7/tau_i).*G_int + (k7/tau_i).*G_b + (k8.*tau_d).*D.G_PL);
%insulin  liver
I_liv = k7*(G_b./(f_I*tau_i*I_PL_b)).*I_PL;
%insulin concentration in remote compartment
D.I_d1 = k9*(I_PL -I_PL_b) - k10*I_d1;

%transport of insulin from plasma to remote compartment
i_rem = k9*(I_PL - I_PL_b);

%rate of change of plasma insulin
D.I_PL = I_pnc - I_liv - i_rem;

%% -- catch an error where the timestep of the integration becomes too small
MINSTEP = 1e-10; %Minimum step

persistent tprev

if isempty(tprev)
    tprev = -inf;
end
timestep = t - tprev;
tprev = t;

if (timestep > 0) && (timestep < MINSTEP)
    error(['Stopped. Time step is too small: ' num2str(timestep)])
end
%% Output of differential equations

dxdt=[D.M_G_gut;D.G_PL;D.G_int;D.I_PL;D.I_d1];

