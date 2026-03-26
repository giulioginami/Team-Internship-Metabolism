function [ngtGroup, igtGroup, t2dmGroup, exGroup] = Classify_DiabetesNIT2(inputGlucose,inputInsulin, Ts)
% Classify_DiabetesNIT2 Classify diabetic condition based on glucose/insulin data realizations
%
% Syntax [ngtGroup, igtGroup, t2gtGroup, oodGroup] 
%         = Classify_DiabetesNIT2GT(inputGlucose,inputInsulin, Ts)%% Purpose% 
%
% Input: time-series data (t=0.0 => i=0)
%   inputGlucose (M realizations × N samples)
%   inputInsulin (M realizations × N samples)
%   Ts: Discrete sample time [s]
%
% Output: Classified diabetes groups 
%   ngtGroup:  Normal Glucose Tolerance Group (P realizations × N samples)
%   igtGroup:  Impaired Glucose Tolerance Group (Q realizations × N samples)
%   t2gtGroup: Type2 Glucose Tolerance Group (R realizations × N samples)
%   exGroup:   Excluded Group (o-o-d, invalid) (S realizations × N samples)
%
% Example usage%  [ngt,igt,t2gt,ood] = Classify_DiabetesNIT2GT(glucoseMatrix, insulinMatrix, 60);

%% ========================================================================
%%  Adhere ADA ruleset 2026 - Glucose Tolerance / Diabetes solely defined by glucose levels
%% ========================================================================

%% ========================================================================
%% HOMA-IR  homeostatic model assessment for insulin resistance: insuline resistance index basal_glucose * basal_insulin    resitivity
%% QUICKI   1/ (log(basal_glucose) +  log(basal_glucose))   sensitivity - preferred method
%% UP_TRESHOLD: 
%% ========================================================================
   % gold standard: 75 g anhydrous glucose dissolved in water

   % HOMA-IR = [(Fasting Insulin (µU/mL)) X (Fasting Glucose (mmol/L))]/22.5
   % QUICKI = 1/[Log (Fasting Insulin, µU/ml) + Log (Fasting Glucose, mg/dl)]

   %% Characterization of OGTT 
   % Matsuda Index (insulin+glucose   basal + average)
   % Insulin AUC / Glucose AUC


end

