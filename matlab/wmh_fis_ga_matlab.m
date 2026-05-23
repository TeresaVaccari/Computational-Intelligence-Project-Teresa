%% WMH segmentation with MATLAB FIS and Genetic Algorithm
% This script uses the FIS defined in wmh_build_fis.m, which contains the
% same rules shown in wmh_fis_matlab_rules.m.

clc
close all
clear all

rng(42)

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir)

dataDir = fullfile(projectRoot, 'data');
maxSlices = 3;
minMaskPixels = 10;
populationSize = 40;
generations = 30;
mutationRate = 0.15;
threshold = 0.5;

fis = wmh_build_fis();
data = load_dataset(dataDir);
trainingSlices = select_training_slices(data.mask, maxSlices, minMaskPixels);

fprintf('Training slices: ')
fprintf('%d ', trainingSlices)
fprintf('\n')

nRules = numel(fis.Rules);
baselineWeights = 0.5 * ones(1, nRules);
baselineDice = evaluate_weights(fis, data, trainingSlices, baselineWeights, threshold);
fprintf('Baseline Dice with equal weights: %.4f\n', baselineDice)

[bestWeights, bestDice, history] = genetic_algorithm( ...
    fis, data, trainingSlices, populationSize, generations, mutationRate, threshold);

fprintf('\nBest rule weights:\n')
for i = 1:numel(bestWeights)
    fprintf('Rule %02d: %.3f\n', i, bestWeights(i))
end
fprintf('\nBest Dice: %.4f\n', bestDice)

plot_result(fis, data, trainingSlices(1), bestWeights, threshold)

%% Local functions

function data = load_dataset(dataDir)
    data.flair = load_npy(fullfile(dataDir, 'FLAIR_dataset.npy'));
    data.mask = load_npy(fullfile(dataDir, 'WMH_masks.npy'));
    data.mean = load_npy(fullfile(dataDir, 'mean_dataset.npy'));
    data.std = load_npy(fullfile(dataDir, 'std_dataset.npy'));
    data.skew = load_npy(fullfile(dataDir, 'skew_dataset.npy'));
    data.kurtosis = load_npy(fullfile(dataDir, 'kurtosis_dataset.npy'));

    data.skew(isnan(data.skew)) = -5;
    data.kurtosis(isnan(data.kurtosis)) = -5;
end

function x = load_npy(filename)
    % Loads .npy files using MATLAB's Python interface.
    npArray = py.numpy.load(filename);
    shape = double(py.array.array('d', npArray.shape));
    flat = double(py.array.array('d', npArray.flatten('F')));
    x = reshape(flat, shape);
end

function xNorm = minmax_normalize(x)
    epsValue = 1e-8;
    xMin = min(x(:));
    xMax = max(x(:));
    xNorm = (x - xMin) ./ (xMax - xMin + epsValue);
end

function features = prepare_slice_features(data, sliceIdx)
    flair = data.flair(:, :, sliceIdx);

    intensity = minmax_normalize(flair);
    localMean = minmax_normalize(data.mean(:, :, sliceIdx));
    localStd = minmax_normalize(data.std(:, :, sliceIdx));
    skewness = minmax_normalize(data.skew(:, :, sliceIdx));
    kurtosis = minmax_normalize(data.kurtosis(:, :, sliceIdx));
    localContrast = minmax_normalize(flair - data.mean(:, :, sliceIdx));

    localMax = movmax(movmax(flair, [2 2], 1), [2 2], 2);
    localMin = movmin(movmin(flair, [2 2], 1), [2 2], 2);
    localRange = minmax_normalize(localMax - localMin);

    features = [
        intensity(:), ...
        localMean(:), ...
        localStd(:), ...
        skewness(:), ...
        kurtosis(:), ...
        localContrast(:), ...
        localRange(:)
    ];
end

function fisWeighted = set_rule_weights(fis, weights)
    fisWeighted = fis;
    for i = 1:numel(weights)
        fisWeighted.Rules(i).Weight = weights(i);
    end
end

function prediction = segment_slice(fis, data, sliceIdx, weights, threshold)
    fisWeighted = set_rule_weights(fis, weights);
    featureMatrix = prepare_slice_features(data, sliceIdx);
    score = evalfis(fisWeighted, featureMatrix);
    prediction = reshape(score >= threshold, size(data.flair(:, :, sliceIdx)));
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

function slices = select_training_slices(mask, maxSlices, minMaskPixels)
    maskPixels = squeeze(sum(sum(mask > 0, 1), 2));
    candidates = find(maskPixels >= minMaskPixels);

    if isempty(candidates)
        [~, order] = sort(maskPixels, 'descend');
        slices = order(1:maxSlices);
    else
        slices = candidates(1:min(maxSlices, numel(candidates)));
    end
end

function meanDice = evaluate_weights(fis, data, sliceIndices, weights, threshold)
    diceValues = zeros(1, numel(sliceIndices));

    for i = 1:numel(sliceIndices)
        sliceIdx = sliceIndices(i);
        prediction = segment_slice(fis, data, sliceIdx, weights, threshold);
        target = data.mask(:, :, sliceIdx) > 0;
        diceValues(i) = dice_score(prediction, target);
    end

    meanDice = mean(diceValues);
end

function [bestWeights, bestDice, history] = genetic_algorithm( ...
    fis, data, sliceIndices, populationSize, generations, mutationRate, threshold)

    nRules = numel(fis.Rules);
    population = rand(populationSize, nRules);
    history = zeros(1, generations);

    for generation = 1:generations
        fitness = zeros(populationSize, 1);

        for i = 1:populationSize
            fitness(i) = evaluate_weights(fis, data, sliceIndices, population(i, :), threshold);
        end

        [fitness, order] = sort(fitness, 'descend');
        population = population(order, :);
        history(generation) = fitness(1);
        fprintf('Generation %03d | best Dice = %.4f\n', generation, fitness(1))

        eliteCount = max(2, floor(populationSize / 5));
        newPopulation = population(1:eliteCount, :);

        while size(newPopulation, 1) < populationSize
            parentA = tournament_select(population, fitness);
            parentB = tournament_select(population, fitness);
            child = crossover(parentA, parentB);
            child = mutate(child, mutationRate);
            newPopulation = [newPopulation; child]; %#ok<AGROW>
        end

        population = newPopulation;
    end

    fitness = zeros(populationSize, 1);
    for i = 1:populationSize
        fitness(i) = evaluate_weights(fis, data, sliceIndices, population(i, :), threshold);
    end

    [bestDice, bestIdx] = max(fitness);
    bestWeights = population(bestIdx, :);
end

function parent = tournament_select(population, fitness)
    tournamentSize = 3;
    idx = randperm(size(population, 1), tournamentSize);
    [~, bestLocalIdx] = max(fitness(idx));
    parent = population(idx(bestLocalIdx), :);
end

function child = crossover(parentA, parentB)
    mask = rand(size(parentA)) < 0.5;
    child = parentA;
    child(~mask) = parentB(~mask);
end

function child = mutate(child, mutationRate)
    mutationMask = rand(size(child)) < mutationRate;
    noise = 0.12 * randn(size(child));
    child = child + mutationMask .* noise;
    child = min(max(child, 0), 1);
end

function plot_result(fis, data, sliceIdx, weights, threshold)
    fisWeighted = set_rule_weights(fis, weights);
    featureMatrix = prepare_slice_features(data, sliceIdx);
    score = evalfis(fisWeighted, featureMatrix);
    scoreImage = reshape(score, size(data.flair(:, :, sliceIdx)));
    prediction = scoreImage >= threshold;
    target = data.mask(:, :, sliceIdx) > 0;
    dice = dice_score(prediction, target);

    figure('Name', 'WMH FIS GA result')
    subplot(1, 4, 1)
    imagesc(data.flair(:, :, sliceIdx)')
    axis image off
    colormap gray
    title('FLAIR')

    subplot(1, 4, 2)
    imagesc(target')
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
