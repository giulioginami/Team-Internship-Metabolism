%% Compare_Prev_New_Japan.m
% TASK D-4 deliverable: one figure that shows the synthetic data improved.
% Overlays, per ADA category, for glucose and insulin:
%   - Japan REAL          (black markers +/- IQR, 5-point OGTT)
%   - PREVIOUS synthetic  (red   median + IQR band)  virtual_population_labelled.mat
%   - NEW synthetic v2    (blue  median + IQR band)  virtual_population_v2_aug_labelled.mat
% plus a bar of the headline mean|SMD| (previous vs new) - the single
% number that proves the improvement. SMD logic is recomputed here so the
% figure is self-contained and independent of run order.
%
% Requires the three labelled .mat files to exist.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; close all;

PREV = 'virtual_population_labelled.mat';
NEW  = 'virtual_population_v2_aug_labelled.mat';
if ~isfile(NEW)   % fall back if step-4 augmentation not run yet
    NEW = 'virtual_population_v2_labelled.mat';
    warning('Using %s (run Augment_AgeBMI_v2 for the age/BMI version).', NEW);
end

P = load(PREV, 'dataset_NGT','dataset_IGT','dataset_T2DM');
N = load(NEW,  'dataset_NGT','dataset_IGT','dataset_T2DM');
load('japan_population_labelled.mat', 'japan_population'); J = japan_population;

cats   = {'NGT','IGT','T2DM'};
jmask  = {J.is_NGT, J.is_IGT, J.is_T2DM};
ogtt_t = [0 30 60 90 120];
prev_ds = {P.dataset_NGT,P.dataset_IGT,P.dataset_T2DM};
new_ds  = {N.dataset_NGT,N.dataset_IGT,N.dataset_T2DM};
cP = [0.80 0.15 0.15]; cN = [0.15 0.35 0.85];

%% ---- headline mean|SMD| (prev & new) -----------------------------------
hd = struct();
for s = 1:2
    DS = prev_ds; if s==2, DS = new_ds; end
    acc = [];
    for vi = 1:2                                  % 1=glucose 2=insulin
        vf = 'glucose_noisy'; if vi==2, vf='insulin_noisy'; end
        for c = 1:3
            tv = DS{c}.time(:)';
            [~,ix] = arrayfun(@(tt) min(abs(tv-tt)), ogtt_t);
            Xv = DS{c}.(vf)(:, ix);
            Xj = J.(vf)(jmask{c}, :);
            for k = 1:5
                a = Xv(:,k); a=a(isfinite(a));
                b = Xj(:,k); b=b(isfinite(b));
                if numel(a)>2 && numel(b)>2
                    acc(end+1) = smd(a,b); %#ok<SAGROW>
                end
            end
        end
    end
    if s==1, hd.prev = mean(abs(acc)); else, hd.new = mean(abs(acc)); end
end
fprintf('Headline mean|SMD|  previous = %.3f  ->  new = %.3f  (%.0f%% closer)\n', ...
    hd.prev, hd.new, 100*(hd.prev-hd.new)/hd.prev);

%% ---- figure ------------------------------------------------------------
figure('Name','D-4: REAL vs PREVIOUS vs NEW synthetic','Color','w', ...
       'Position',[60 60 1500 820]);
for col = 1:2
    vf = 'glucose_noisy'; ylab='Glucose (mmol/L)';
    if col==2, vf='insulin_noisy'; ylab='Insulin (mU/L)'; end
    for r = 1:3
        ax = subplot(2,3,(col-1)*3+r); hold(ax,'on');
        % previous & new bands
        hP = band(ax, prev_ds{r}.time, prev_ds{r}.(vf), cP);
        hN = band(ax, new_ds{r}.time,  new_ds{r}.(vf),  cN);
        % Japan markers
        dj  = J.(vf)(jmask{r}, :);
        mj  = median(dj,1); q1=prctile(dj,25,1); q3=prctile(dj,75,1);
        hJ  = errorbar(ax, ogtt_t, mj, mj-q1, q3-mj, 'o--k', ...
              'MarkerFaceColor','k','MarkerSize',6,'LineWidth',1.5,'CapSize',8);
        xlim(ax,[-5 140]); grid(ax,'on'); set(ax,'Layer','top');
        xlabel(ax,'Time (min)','FontWeight','bold');
        ylabel(ax,ylab,'FontWeight','bold');
        title(ax,sprintf('%s  (prev n=%d, new n=%d, Japan n=%d)', ...
              cats{r}, prev_ds{r}.n, new_ds{r}.n, sum(jmask{r})));
        if col==1 && r==1
            legend([hJ hP hN], {'Japan (real, med\pmIQR)', ...
                'Previous synthetic (med, IQR)','New synthetic v2 (med, IQR)'}, ...
                'Location','northwest','FontSize',9);
        end
    end
end
sgtitle(sprintf(['REAL vs PREVIOUS vs NEW synthetic   |   headline mean|SMD|: ' ...
    '%.3f \\rightarrow %.3f  (%.0f%% closer to real)'], ...
    hd.prev, hd.new, 100*(hd.prev-hd.new)/hd.prev), 'FontSize',14,'FontWeight','bold');

% small inset bar of the headline
axes('Position',[0.46 0.46 0.09 0.10]); bar([hd.prev hd.new]); grid on;
set(gca,'XTickLabel',{'prev','new'}); ylabel('mean|SMD|'); title('headline');

%% ---- local SMD / band --------------------------------------------------
function d = smd(a,b)
    sp = sqrt(((numel(a)-1)*var(a)+(numel(b)-1)*var(b))/max(numel(a)+numel(b)-2,1));
    if sp==0, d=0; else, d=(mean(a)-mean(b))/sp; end
end
function h = band(ax,t,X,clr)
    t=t(:)'; m=median(X,1); q1=prctile(X,25,1); q3=prctile(X,75,1);
    fill(ax,[t fliplr(t)],[q1 fliplr(q3)],clr,'EdgeColor','none','FaceAlpha',0.20);
    h=plot(ax,t,m,'-','Color',clr,'LineWidth',2.2);
end
