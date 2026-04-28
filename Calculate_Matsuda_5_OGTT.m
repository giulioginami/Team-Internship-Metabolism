function matsuda = Calculate_Matsuda_5_OGTT(glucose5, insulin5)
% Matsuda Index (insulin+glucose   basal + average)
%   dynamic Matsuda et al. 1999: ~2.5
%   Katharina Lechner et al. 2021:
%       Cutoff normal > 4.0     
%       Pheripheral IR <= 4.0
    G_mg_dl = 18.0156 .* glucose5;
    I_muU_ml = insulin5;
    matsuda = 1e4 ./ sqrt( G_mg_dl(:, 1) .* I_muU_ml(:, 1) .* ...
        ( 15.*sum(G_mg_dl(:, [1,5]), 2) + 30.*sum(G_mg_dl(:, [2,3,4]), 2) )/120 .* ...
        ( 15.*sum(I_muU_ml(:, [1,5]), 2) + 30.*sum(I_muU_ml(:, [2,3,4]), 2) )/120 );
end