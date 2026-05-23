%% WMH rule-weight optimization using MATLAB ga
% This script follows the same structure as the ObjFunGA examples:
%
%   FitnessFunction = @(chromosome) ObjFunGA_WMH(...)
%   [x, fval] = ga(FitnessFunction, numberOfVariables, ...)
%
% Here, each chromosome is a vector of fuzzy rule weights.

clc
close all
clear all

rng(42)

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir)

threshold = 0.5;
populationSize = 40;
generations = 30;

fis = wmh_fis_rules(false);
nRules = numel(fis.Rules);

[featureMatrices, manualMasks, imageShapes, flairSlices, sourceFiles] = load_preprocessed_features(projectRoot);

baselineWeights = 0.5 * ones(1, nRules);
baselineError = ObjFunGA_WMH(baselineWeights, fis, featureMatrices, manualMasks, imageShapes, threshold);
baselineDice = 1 - baselineError;

FitnessFunction = @(chromosome) ObjFunGA_WMH( ...
    chromosome, fis, featureMatrices, manualMasks, imageShapes, threshold);

numberOfVariables = nRules;
lowerBounds = zeros(1, nRules);
upperBounds = ones(1, nRules);

% GA options written in the same style as the provided examples.
opts.InitialPopulationRange = [lowerBounds; upperBounds];
opts.SelectionFcn = @selectionstochunif; % parent selection with stochastic uniform method
opts.PopulationSize = populationSize;
opts.Generations = generations;
opts.CrossoverFraction = 0.8; % fraction of the population at the next generation that the crossover function creates
opts.MutationFcn = @mutationadaptfeasible; % mutation function, randomly modifies some weights in the chromosome
opts.EliteCount = max(2, floor(populationSize / 5)); % number of elite population members, preserved without modification in the next generation
opts.PlotFcns = @gaplotbestf;

[bestWeights, bestError] = ga( % ga as in example2.m
    FitnessFunction,
    numberOfVariables,
    [], [], [], [],
    lowerBounds,
    upperBounds,
    [],
    opts);

bestDice = 1 - bestError;

fprintf('\nBest rule weights:\n')
for i = 1:numel(bestWeights)
    fprintf('Rule %02d: %.3f\n', i, bestWeights(i))
end
fprintf('\nBest Dice: %.4f\n', bestDice)

plot_result(fis, bestWeights, featureMatrices{1}, manualMasks{1}, imageShapes{1}, flairSlices{1}, threshold)

outputDir = fullfile(projectRoot, 'matlab_ga_output');
if ~exist(outputDir, 'dir')
    mkdir(outputDir)
end

save(fullfile(outputDir, 'ga_optimized_rule_weights.mat'), ...
    'bestWeights', 'bestDice', 'baselineDice', 'threshold', 'sourceFiles')


%% Local helper functions

function [featureMatrices, manualMasks, imageShapes, flairSlices, sourceFiles] = load_preprocessed_features(projectRoot)
featureDir = fullfile(projectRoot, 'matlab_preprocessed_features');
files = dir(fullfile(featureDir, 'preprocessed_features_slice_*.mat'));

if isempty(files)
    error('No preprocessed feature .mat files found. Run export_preprocessed_features_for_matlab.py first.')
end

featureMatrices = cell(1, numel(files));
manualMasks = cell(1, numel(files));
imageShapes = cell(1, numel(files));
flairSlices = cell(1, numel(files));
sourceFiles = strings(1, numel(files));

for i = 1:numel(files)
    filePath = fullfile(files(i).folder, files(i).name);
    loaded = load(filePath);

    featureMatrices{i} = loaded.featureMatrix;
    manualMasks{i} = loaded.manualMask > 0;
    imageShapes{i} = double(loaded.imageShape);
    flairSlices{i} = loaded.flairSlice;
    sourceFiles(i) = string(filePath);
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
scoreImage = reshape(score, imageShape(1), imageShape(2));
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
