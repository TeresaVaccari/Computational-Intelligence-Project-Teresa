%% WMH rule-weight optimization using MATLAB ga

clc
close all
clearvars -except populationSize generations threshold

rng(42)
warningState = warning;
cleanupWarning = onCleanup(@() warning(warningState));
warning('off', 'all')

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir)



    threshold = 0.65;

    populationSize = 10;

    generations = 5;


fis = wmh_fis_rules(false);
nRules = numel(fis.Rules);



[featureMatrices, manualMasks, imageShapes, flairSlices, sourceFiles] = load_preprocessed_features(projectRoot);



fitnessFunction = @(chromosome) ObjFunGA_WMH( ...
    chromosome, fis, featureMatrices, manualMasks, imageShapes, threshold);



numberOfVariables = nRules + 1;             %rule weights + threshold
lowerBounds = [zeros(1, nRules), 0.05];
upperBounds = [ones(1, nRules), 0.95];

opts = optimoptions('ga', ...
    'PopulationSize', populationSize, ...
    'MaxGenerations', generations, ...
    'CrossoverFraction', 0.8, ...
    'MutationFcn', @mutationadaptfeasible, ...
    'EliteCount', max(2, floor(populationSize / 5)));

[bestChromosome, bestError] = ga( ...
    fitnessFunction, ...
    numberOfVariables, ...
    [], [], [], [], ...
    lowerBounds, ...
    upperBounds, ...
    [], ...
    opts);

bestWeights = bestChromosome(1:nRules);
bestThreshold = bestChromosome(end);
bestDice = 1 - bestError;

fprintf('\nBest rule weights:\n')
for i = 1:numel(bestWeights)
    fprintf('Rule %02d: %.3f\n', i, bestWeights(i))
end
fprintf('Best threshold: %.3f\n', bestThreshold)
fprintf('Best Dice: %.4f\n', bestDice)

plot_result(fis, bestWeights, featureMatrices{1}, manualMasks{1}, imageShapes{1}, flairSlices{1}, bestThreshold)

outputDir = fullfile(projectRoot, 'matlab_ga_output');
if ~exist(outputDir, 'dir')
    mkdir(outputDir)
end

save(fullfile(outputDir, 'ga_optimized_rule_weights.mat'), ...
    'bestWeights', 'bestThreshold', 'bestDice', 'threshold', 'sourceFiles')


function [featureMatrices, manualMasks, imageShapes, flairSlices, sourceFiles] = load_preprocessed_features(projectRoot)
featureDir = fullfile(projectRoot, 'matlab_preprocessed_features');
files = dir(fullfile(featureDir, 'preprocessed_features_slice_*.mat'));


featureMatrices = cell(1, numel(files));
manualMasks = cell(1, numel(files));
imageShapes = cell(1, numel(files));
flairSlices = cell(1, numel(files));
sourceFiles = strings(1, numel(files));

for i = 1:numel(files)
    loaded = load(fullfile(files(i).folder, files(i).name));
    featureMatrices{i} = loaded.featureMatrix;
    manualMasks{i} = loaded.manualMask > 0;
    imageShapes{i} = double(loaded.imageShape);
    flairSlices{i} = loaded.flairSlice;
    sourceFiles(i) = string(fullfile(files(i).folder, files(i).name));
end
end


function fisWeighted = set_rule_weights(fis, ruleWeights)
fisWeighted = fis;
for i = 1:numel(ruleWeights)
    fisWeighted.Rules(i).Weight = ruleWeights(i);
end
end


function dice = dice_score(prediction, target)
prediction = logical(prediction);
target = logical(target);
intersection = sum(prediction(:) & target(:));
denom = sum(prediction(:)) + sum(target(:));

if denom == 0
    dice = 1;
else
    dice = 2 * intersection / denom;
end
end


function plot_result(fis, ruleWeights, featureMatrix, manualMask, imageShape, flairSlice, threshold)
fisWeighted = set_rule_weights(fis, ruleWeights);
score = evalfis(fisWeighted, featureMatrix);
scoreImage = vector_to_image(score, imageShape);
prediction = scoreImage >= threshold;
dice = dice_score(prediction, manualMask);

figure('Name', 'GA optimized WMH segmentation')

subplot(1, 4, 1)
imagesc(flairSlice')
axis image off
colormap gray
title('FLAIR')

subplot(1, 4, 2)
imagesc(manualMask')
axis image off
title('Manual mask')

subplot(1, 4, 3)
imagesc(scoreImage')
axis image off
title('FIS score')

subplot(1, 4, 4)
imagesc(prediction')
axis image off
title(sprintf('Prediction | Dice=%.3f', dice))
end


function image = vector_to_image(vector, imageShape)
image = reshape(vector, imageShape(2), imageShape(1))';
end
