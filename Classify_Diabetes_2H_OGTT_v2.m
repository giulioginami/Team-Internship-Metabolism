function [ngtGroup, predmGroup, t2dmGroup, exGroup] = Classify_Diabetes_2H_OGTT_v2(inputGlucose,inputInsulin)
% Classify_Diabetes_2H_OGTT_v2  -  STEP 3 corrected classifier.
%
% IDENTICAL to Classify_Diabetes_2H_OGTT.m EXCEPT the T2DM rule is the
% standard ADA logical OR instead of AND:
%
%   original : ADA_T2DM = (G_0 >= 7.0)  &  (G_2h >= 11.1)
%   v2 (here): ADA_T2DM = (G_0 >= 7.0)  |  (G_2h >= 11.1)
%
% WHY: standard ADA criteria are OR (fasting OR 2-h), and the Japan study
% labels its T2DM patients by the same OR convention. Step-2 marginals
% showed Japan T2DM has median fasting G_b = 5.78 (p95 = 7.65), i.e. most
% real T2DM patients have G_0 < 7.0 and would be MISSED by the AND rule.
% Using AND made virtual T2DM artificially fasting-hyperglycaemic
% (the +3.58 SMD at t=0 in step 1) and, with the re-prioredd realistic
% G_b bound, would yield almost no T2DM at all.
%
% The original Classify_Diabetes_2H_OGTT.m is left untouched; switching
% back is just a matter of calling the original function instead.
%
% Input/Output: identical signature to the original.
%   inputGlucose / inputInsulin : [M realizations x 5] (t = 0,30,60,90,120)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    G_0  = inputGlucose(:, 1);
    G_2h = inputGlucose(:, 5);
    G_max = max(inputGlucose, [], 2);
    G_min = min(inputGlucose, [], 2);

    % ----- ADA ruleset (T2DM = OR; this is the only change vs original) -----
    ADA_T2DM  = (G_0 >= 7.0 | G_2h >= 11.1)               &  G_max < 25.0;
    ADA_PREDM = (G_0 < 7.0) & (G_2h >= 7.8 & G_2h < 11.1) &  G_min > 2.0;
    ADA_NGT   = (G_0 < 7.0) & (G_2h < 7.8)                &  G_min > 3.9;

    T2DM_set  = ADA_T2DM;
    NGT_set   = ADA_NGT;
    PREDM_set = ADA_PREDM;

    % Curve-sanity filters (kept from the original)
    % 1) NGT must not peak above 10 mmol/L
    NGT_set  = NGT_set  & (G_max <= 10.0);
    % 4) drop extreme fasting (data-entry / non-physiological)
    T2DM_set = T2DM_set & (G_0 < 11.0);

    % Enforce mutual exclusivity with explicit ADA precedence: T2DM > IGT > NGT
    NGT_set   = NGT_set   & ~T2DM_set;
    PREDM_set = PREDM_set & ~T2DM_set & ~NGT_set;

    out_distribution_set = ~(T2DM_set | NGT_set | PREDM_set);

    ngtGroup   = NGT_set;
    predmGroup = PREDM_set;
    t2dmGroup  = T2DM_set;
    exGroup    = out_distribution_set;
end
