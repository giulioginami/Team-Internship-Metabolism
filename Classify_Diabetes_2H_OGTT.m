function [ngtGroup, predmGroup, t2dmGroup, exGroup] = Classify_Diabetes_2H_OGTT(inputGlucose,inputInsulin)
% Classify_Diabetes_2H_OGTT Classify diabetic condition based on glucose/insulin data realizations
%
% Syntax [ngtGroup, igtGroup, t2gtGroup, oodGroup] 
%         = Classify_DiabetesNIT2GT(inputGlucose,inputInsulin) 
%
% Input: time-series data (t=0.0 => i=0)
%   inputGlucose (M realizations × N samples)
%   inputInsulin (M realizations × N samples)
%   Ts: Discrete sample time [s]
%
% Output: Classified diabetes groups 
%   ngtGroup:  Normal Glucose Tolerance Group (P realizations × N samples)
%   predmGroup:Prediabetes Group (Q realizations × N samples)
%   t2gtGroup: Type2 Glucose Tolerance Group (R realizations × N samples)
%   exGroup:   Excluded Group (o-o-d, invalid) (S realizations × N samples)
%
% Example usage%  [ngt,prd,t2gt,ood] = Classify_Diabetes(glucoseMatrix, insulinMatrix);

%% ========================================================================
%%  Adhere ADA ruleset 2026 - Glucose Tolerance / Diabetes solely defined by glucose levels
%% ========================================================================
   % known standard: 5-point 2H OGTT 75 g anhydrous glucose dissolved in water
   %              t =[0, 30, 60, 90, 120]  [min]
   %        ensure standard!!
%%======================================================================

%%======================================================================
%% Classification Methodology
%%======================================================================
% Leading diabetes Metric:    75g glucose OGTT (G_2h)
% Alternative Sensitivity Metric: 
%   !(only G_fasting not the best differentiator between groups)
%   however, take still into consideration.


%% T2DM
% Leading Metric:     75g glucose OGTT; G_0 >= 7.0; G_2h >= 11.1
% Alternative Metric: QUICKI <= 0.33  or Matsuda <= 4.0   

%% NGT
% Leading Metric:     75g glucose OGTT; G_0 < 7.0; G_2h < 7.8
% Alternative Metric: QUICKI >= 0.348 or Matsuda > 4.0   

%% preDM
% Leading Metric:     75g glucose OGTT; 5.6 < G_0 < 7.0; 7.8 <= G_2h < 11.1
% Alternative Metric: QUICKI <= 0.33  or Matsuda <= 4.0  


    %% ========================================================================
    % Remove nan entries
    %nan = find(sum(isnan([inputGlucose inputInsulin]), 1));
    %inputGlucose(nan, :) = [];
    %inputInsulin(nan, :) = [];

%% ========================================================================
% Apply methodology
% Precedence: ADA > QUICKI v MATSUDA 
%% =========================================================================
    G_0 = inputGlucose(:, 1);
    G_2h = inputGlucose(:, 5);
    G_max = max(inputGlucose, [], 2);
    G_min = min(inputGlucose, [], 2);                       % for clamping boundaries
    ADA_T2DM = (G_0 >=7.0  |  G_2h >= 11.1)                 &  G_max < 25.0; %bump up for now %16.7; % hyperglycemia 
    ADA_PREDM= (G_0 < 7.0) & (G_2h >= 7.8 & G_2h < 11.1)    &  G_min > 2.0;  %bump down for now %3.9 ; % hypoglycemia 2~3  peak -> lower through T2 stage update through time  
    ADA_NGT  = (G_0 < 7.0) & (G_2h < 7.8)                   &  G_min > 3.9; 
    
    % QUICKI = Calculate_QUICKI(inputGlucose(:, 1), inputInsulin(:, 1))
    % QUICKI_RESISTANT = QUICKI < 0.33             &  QUICKI>0.2%0.30; % glucose intolerance
    % QUICKI_HEALTHY = QUICKI >= 0.33             &  QUICKI<0.6%=0.43; % glucose hypersensitivity %300-400 insulin
    % 
    % Matsuda = Calculate_Matsuda_5_OGTT(inputGlucose, inputInsulin)
    % Matsuda_RESISTANT = Matsuda <= 4             &  Matsuda>0.80%1.0; % 
    % Matsuda_HEALTHY = Matsuda > 4                &  Matsuda<8;  % 
    
    % select initial in distribution set
    % in_distribution = (QUICKI_RESISTANT | QUICKI_HEALTHY ) & (Matsuda_RESISTANT | Matsuda_HEALTHY);
    T2DM_set = ADA_T2DM;     % & in_distribution;
    NGT_set = ADA_NGT;       % & in_distribution & QUICKI_HEALTHY & Matsuda_HEALTHY;
    % resistant_NGT = ADA_NGT & in_distribution & (QUICKI_RESISTANT | Matsuda_RESISTANT);
    PREDM_set = ADA_PREDM;   % resistant_NGT | ADA_PREDM & in_distribution & (QUICKI_RESISTANT | Matsuda_RESISTANT);
    out_distribution_set = ~(T2DM_set | NGT_set | PREDM_set);
   
    ngtGroup = NGT_set;
    predmGroup = PREDM_set;
    t2dmGroup = T2DM_set;
    exGroup = out_distribution_set;
end


%% ========================================================================
%% HOMA-IR  homeostatic model assessment for insulin resistance: insuline resistance index basal_glucose * basal_insulin    resitivity
%% QUICKI   1/ (log(basal_Insulin) +  log(basal_glucose))   sensitivity - preferred method

%% Insulin sensitivity assessment
%%======================================================================
%% Characterization static response via FPG/FPI (Insulin s)
%%    useful for determining Impaired Fasting Glucose IFG condition
% HOMA-IR = [(Fasting Insulin (µU/mL)) X (Fasting Glucose (mmol/L))]/22.5
% QUICKI = 1/[Log (Fasting Insulin, µU/ml) + Log (Fasting Glucose, mg/dl)]

%% Characterization dynamic repsonse via OGTT (Leading metric)
%%    useful for determining Impaired Glucose Tolerance GT condition
% Matsuda Index (insulin+glucose   basal + average)
% Insulin AUC / Glucose AUC

% function matsuda = Calculate_Matsuda_5_OGTT(glucose5, insulin5)
%     G_mg_dl = 18.0156 .* glucose5;
%     I_muU_ml = insulin5;
%     matsuda = 1e4 ./ sqrt( G_mg_dl(:, 1) .* I_muU_ml(:, 1) .* ...
%         ( 15.*sum(G_mg_dl(:, [1,5]), 2) + 30.*sum(G_mg_dl(:, [2,3,4]), 2) )/120 .* ...
%         ( 15.*sum(I_muU_ml(:, [1,5]), 2) + 30.*sum(I_muU_ml(:, [2,3,4]), 2) )/120 );
% end
% 
% function quicki = Calculate_QUICKI(FPG, FPI)
%     G0_mg_dl = 18.0156 * FPG;
%     I0_muU_ml = FPI;
%     quicki = 1.0 ./ ( log10(I0_muU_ml) +  log10(G0_mg_dl));
% end
