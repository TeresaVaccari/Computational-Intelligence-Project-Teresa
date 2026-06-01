clear variables;
close all;
clc;

rng(42)
warningState = warning;
cleanupWarning = onCleanup(@() warning(warningState));
warning('off', 'all')

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir) % add matlab path to script

data = load(fullfile(scriptDir, 'preprocessed_features_slice_31.mat'));

featureMatrix = data.feature_matrix; % shape 28086 x 9
gold_standard = data.manual_mask == 1; % manual mask shape 151 x 186
gold_vector = data.manual_mask_vector == 1; % vector column corresponding to rows of featureMatrix

% fill parameter struct: each feature is a column of featureMatrix
param.intensity = featureMatrix(:, 1); 
param.mean = featureMatrix(:, 2);
param.std = featureMatrix(:, 3);
param.skew = featureMatrix(:, 4);
param.kurtosis = featureMatrix(:, 5);
param.contrast = featureMatrix(:, 6);
param.range = featureMatrix(:, 7);
param.x = featureMatrix(:, 8);
param.y = featureMatrix(:, 9);
param.gold = gold_vector(:);
param.threshold = 0.5;
param.thresholds = 0:0.01:1;

lesion_idx = find(param.gold == 1); % lesion pixels
healthy_idx = find(param.gold == 0);

% randomly take a sample of healthy pixel (4 times the number of lesion pixel) to balance the dataset
healthy_idx = healthy_idx(randperm(numel(healthy_idx), 4 * numel(lesion_idx)));

% combine lesion and healthy pixel indices for training
param.pixel_idx = [lesion_idx; healthy_idx]; 

% genetic algorithm
fis_base = fis_rules(true);
numberOfVariables = length(fis_base.Rules);

FitnessFunction = @(chromosome)ObjFunGA_WMH(chromosome, fis_base, param);

opts.InitialPopulationRange = [zeros(1, numberOfVariables); ones(1, numberOfVariables)];
opts.SelectionFcn = @selectionroulette;
opts.PopulationSize = 20;
opts.Generations = 10;
opts.CrossoverFraction = 0.8;
opts.MutationFcn = @mutationadaptfeasible;
opts.EliteCount = 1;
opts.Display = 'iter';

% execution of ga
% x is best solution (weights for each fuzzy rule)
% fval is best fitness (1 - Dice), has to be minimized
% exitflag and output give info about the optimization process
% final_pop and scores give info about the population at the end of the optimization
[x, fval, exitflag, output, final_pop, scores] = ga(FitnessFunction, numberOfVariables, [], [], [], [], zeros(1, numberOfVariables), ones(1, numberOfVariables), [], opts);

% result
fprintf('\nBest fitness: %.4f\n', fval);
fprintf('Rule weights:\n');
for r = 1:numberOfVariables
    fis_base.Rules(r).Weight = x(r);
    fprintf('  Rule %02d: %.4f\n', r, x(r));
end


fis_scores = evalfis(fis_base, featureMatrix); % apply fuzzy system on all pixel of the slice 
dice_scores = zeros(size(param.thresholds)); % save dice for each treshold

% create prediction 0,1 for each threshold
for t = 1:length(param.thresholds)
    treshold_prediction = fis_scores >= param.thresholds(t);
    % compute dice score for each threshold
    denom = sum(treshold_prediction(:)) + sum(param.gold(:));

    if denom == 0
        dice_scores(t) = 1; % if there are no wmh prediction and no real wmh, we consider the dice score 1
    else
        dice_scores(t) = 2 * sum(treshold_prediction(:) & param.gold(:)) / denom;
    end
end

[dice_score, best_idx] = max(dice_scores); % keep best dice score and corresponding threshold idx
best_threshold = param.thresholds(best_idx); % threshold corresponding to treshold idx, found above
% create final prediction with best threshold
prediction = reshape(fis_scores >= best_threshold, size(gold_standard, 2), size(gold_standard, 1))';

fprintf('Training Dice from GA: %.4f\n', 1 - fval);
fprintf('Best threshold: %.2f\n', best_threshold);
fprintf('Dice score: %.4f\n', dice_score);

figure;
subplot(1, 3, 1), imagesc(data.intensity_map'), axis image off, colormap gray, title('FLAIR')
subplot(1, 3, 2), imagesc(gold_standard'), axis image off, title('Manual mask')
subplot(1, 3, 3), imagesc(prediction'), axis image off, title('Predicted mask')

figure;
plot(param.thresholds, dice_scores, 'LineWidth', 1.5)
xline(best_threshold, '--r', sprintf('Best threshold = %.2f', best_threshold))
hold on
plot(best_threshold, dice_score, 'ro', 'MarkerFaceColor', 'r')
hold off
xlabel('Threshold')
ylabel('Dice')
title('Dice vs Threshold')
grid on

save(fullfile(scriptDir, 'ga_optimized_rule_weights.mat'),'x', 'fval', 'best_threshold', 'dice_score', 'exitflag', 'output', 'final_pop', 'scores')
