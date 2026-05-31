clear variables;
close all;
clc;

rng(42)
warningState = warning;
cleanupWarning = onCleanup(@() warning(warningState));
warning('off', 'all')

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir)

data = load(fullfile(projectRoot, ...
    'matlab_preprocessed_features', 'preprocessed_features_slice_31.mat'));

featureMatrix = data.feature_matrix;
gold_standard = data.manual_mask > 0;
gold_vector = data.manual_mask_vector > 0;

%% Fill parameter struct
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

lesion_idx = find(param.gold == 1);
healthy_idx = find(param.gold == 0);
healthy_idx = healthy_idx(randperm(numel(healthy_idx), 4 * numel(lesion_idx)));
param.pixel_idx = [lesion_idx; healthy_idx];

%% GA
fis_base = fis_rules(false);
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

[x, fval, exitflag, output, final_pop, scores] = ga( ...
    FitnessFunction, numberOfVariables, [], [], [], [], ...
    zeros(1, numberOfVariables), ones(1, numberOfVariables), [], opts);

%% Result
fprintf('\nBest fitness: %.4f\n', fval);
fprintf('Rule weights:\n');
for r = 1:numberOfVariables
    fis_base.Rules(r).Weight = x(r);
    fprintf('  Rule %02d: %.4f\n', r, x(r));
end

probabilities = evalfis(fis_base, featureMatrix);
dice_scores = zeros(size(param.thresholds));

for t = 1:length(param.thresholds)
    temp_prediction = probabilities >= param.thresholds(t);
    denom = sum(temp_prediction(:)) + sum(param.gold(:));

    if denom == 0
        dice_scores(t) = 1;
    else
        dice_scores(t) = 2 * sum(temp_prediction(:) & param.gold(:)) / denom;
    end
end

[dice_score, best_idx] = max(dice_scores);
best_threshold = param.thresholds(best_idx);
prediction = reshape(probabilities >= best_threshold, size(gold_standard, 2), size(gold_standard, 1))';

fprintf('Training Dice from GA: %.4f\n', 1 - fval);
fprintf('Best threshold: %.2f\n', best_threshold);
fprintf('Dice score: %.4f\n', dice_score);

figure;
subplot(1, 3, 1), imagesc(data.intensity_map'), axis image off, colormap gray, title('FLAIR')
subplot(1, 3, 2), imagesc(gold_standard'), axis image off, title('Manual mask')
subplot(1, 3, 3), imagesc(prediction'), axis image off, title('Predicted mask')

save(fullfile(projectRoot, 'matlab_ga_output', 'ga_optimized_rule_weights.mat'), ...
    'x', 'fval', 'best_threshold', 'dice_score', 'exitflag', 'output', ...
    'final_pop', 'scores')
