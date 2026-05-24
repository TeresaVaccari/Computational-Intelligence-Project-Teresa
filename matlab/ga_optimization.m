%% WMH rule-weight optimization using MATLAB ga

clc
close all
clear all

rng(42)
warning('off', 'all')

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir)

threshold = 0.65;
populationSize = 10;
generations = 5;

fis_base = fis_rules();

featureDir = fullfile(projectRoot, 'matlab_preprocessed_features');
files = dir(fullfile(featureDir, 'preprocessed_features_slice_*.mat'));

featureMatrices = cell(1, numel(files));
manualMasks = cell(1, numel(files));
imageShapes = cell(1, numel(files));
flairSlices = cell(1, numel(files));

for i = 1:numel(files)
    loaded = load(fullfile(files(i).folder, files(i).name));
    featureMatrices{i} = loaded.featureMatrix;
    manualMasks{i} = loaded.manualMask > 0;
    imageShapes{i} = double(loaded.imageShape);
    flairSlices{i} = loaded.flairSlice;
end

param.featureMatrices = featureMatrices;
param.manualMasks = manualMasks;

numberOfVariables = length(fis_base.Rules);
FitnessFunction = @(chromosome)ObjFunGA_WMH(chromosome, fis_base, param);

lowerBounds = zeros(1, numberOfVariables);
upperBounds = ones(1, numberOfVariables);

opts.PopulationSize = populationSize;
opts.Generations = generations;
opts.CrossoverFraction = 0.8;
opts.MutationFcn = @mutationadaptfeasible;
opts.EliteCount = max(2, floor(populationSize / 5));

[x, fval, exitflag, output, final_pop, scores] = ga(FitnessFunction, numberOfVariables, [], [], [], [], lowerBounds, upperBounds, [], opts);

bestWeights = x;
bestThreshold = threshold;

fprintf('\nBest rule weights:\n')
for i = 1:numel(bestWeights)
    fprintf('Rule %02d: %.3f\n', i, bestWeights(i))
end
fprintf('Best threshold: %.3f\n', bestThreshold)
fprintf('Best error: %.4f\n', fval)

fisWeighted = fis_base;
for i = 1:numel(bestWeights)
    fisWeighted.Rules(i).Weight = bestWeights(i);
end

score = evalfis(fisWeighted, featureMatrices{1});
scoreImage = reshape(score, imageShapes{1}(2), imageShapes{1}(1))';
prediction = scoreImage >= bestThreshold;

figure('Name', 'GA optimized WMH segmentation')

subplot(1, 4, 1)
imagesc(flairSlices{1}')
axis image off
colormap gray
title('FLAIR')

subplot(1, 4, 2)
imagesc(manualMasks{1}')
axis image off
title('Manual mask')

subplot(1, 4, 3)
imagesc(scoreImage')
axis image off
title('FIS score')

subplot(1, 4, 4)
imagesc(prediction')
axis image off
title('Prediction')
