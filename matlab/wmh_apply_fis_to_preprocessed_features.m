%% Apply MATLAB fuzzy rules to preprocessed features exported from Python
% First run:
%   python export_preprocessed_features_for_matlab.py
%
% Then run this MATLAB script.

clc
close all
clear all

scriptDir = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptDir);
addpath(scriptDir)

inputFile = fullfile(projectRoot, 'matlab_preprocessed_features', 'preprocessed_features_slice_31.mat');

if ~isfile(inputFile)
    files = dir(fullfile(projectRoot, 'matlab_preprocessed_features', 'preprocessed_features_slice_*.mat'));
    if isempty(files)
        error('No preprocessed feature .mat file found. Run export_preprocessed_features_for_matlab.py first.')
    end
    inputFile = fullfile(files(1).folder, files(1).name);
end

loaded = load(inputFile);

fis = wmh_build_fis();

featureMatrix = loaded.featureMatrix;
score = evalfis(fis, featureMatrix);

imageShape = double(loaded.imageShape);
scoreImage = reshape(score, imageShape(1), imageShape(2));
prediction = scoreImage >= 0.5;
manualMask = loaded.manualMask > 0;
dice = dice_score(prediction, manualMask);

outputDir = fullfile(projectRoot, 'matlab_fis_output');
if ~exist(outputDir, 'dir')
    mkdir(outputDir)
end

save(fullfile(outputDir, 'matlab_fis_output.mat'), ...
    'scoreImage', 'prediction', 'manualMask', 'dice', 'inputFile')

fprintf('Loaded preprocessed features from: %s\n', inputFile)
fprintf('Feature matrix size: %d x %d\n', size(featureMatrix, 1), size(featureMatrix, 2))
fprintf('Dice with threshold 0.5: %.4f\n', dice)
fprintf('Saved MATLAB FIS output in: %s\n', fullfile(outputDir, 'matlab_fis_output.mat'))

figure('Name', 'MATLAB FIS output from Python preprocessing')
subplot(1, 4, 1)
imagesc(loaded.flairSlice')
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
