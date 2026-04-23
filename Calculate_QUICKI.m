function quicki = Calculate_QUICKI(FPG, FPI)
%% QUICKI   1/ (log(basal_Insulin) +  log(basal_glucose))   sensitivity - preferred method
%   static QUICKI 
%       Healthy QUICKI REF Maryam Tohidi et al. 2014:
%           [0.33–0.42], median 0.37, IQR = [0.35–0.39]    
%       Hisayo Yokoyama et al. 2003: 
%           moderately obese T2 0.338 ± 0.030 
%           normal-weight    T2 0.371 ± 0.037
%           healthy             0.389 ± 0.041   respectively all P < 0.0001 
%                             => [0.348 - 0.43]
%   

    G0_mg_dl = 18.0156 * FPG;
    I0_muU_ml = FPI;
    quicki = 1.0 ./ ( log10(I0_muU_ml) +  log10(G0_mg_dl));
end
