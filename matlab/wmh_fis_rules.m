function fis = wmh_fis_rules(showPlots)
%% WMH Fuzzy Inference System
% FIS written with the same MATLAB style used in ex2.m:
% mamfis -> addInput -> addMF -> addOutput -> addRule.
%
% All input features are normalized in [0, 1].


fis = mamfis('Name', 'WMH_FIS'); % create a Mamdani FIS named 'WMH_FIS'

%% inputs
% 1. Pixel intensity from the FLAIR image.
fis = addInput(fis, [0 1], 'Name', 'intensity');
fis = addMF(fis, 'intensity', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'intensity', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'intensity', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% 2. Local mean feature, already computed with a sliding window.
fis = addInput(fis, [0 1], 'Name', 'local_mean');
fis = addMF(fis, 'local_mean', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_mean', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'local_mean', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% 3.  Local standard deviation feature, already computed with a sliding window.
fis = addInput(fis, [0 1], 'Name', 'local_std');
fis = addMF(fis, 'local_std', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_std', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'local_std', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% 4. Skewness feature, already computed with a sliding window.
fis = addInput(fis, [0 1], 'Name', 'skewness');
fis = addMF(fis, 'skewness', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'skewness', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'skewness', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% 5. Kurtosis feature, already computed with a sliding window.
fis = addInput(fis, [0 1], 'Name', 'kurtosis');
fis = addMF(fis, 'kurtosis', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'kurtosis', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'kurtosis', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% 6. Local contrast: intensity - local_mean.
fis = addInput(fis, [0 1], 'Name', 'local_contrast');
fis = addMF(fis, 'local_contrast', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_contrast', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'local_contrast', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% 7. Local range: max(window) - min(window), computed with a 5x5 window.
fis = addInput(fis, [0 1], 'Name', 'local_range');
fis = addMF(fis, 'local_range', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_range', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'local_range', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

%% output is the degree of belonging to the WMH class
fis = addOutput(fis, [0 1], 'Name', 'WMH');
fis = addMF(fis, 'WMH', 'trimf', [0 0 0.40], 'Name', 'non_wmh');
fis = addMF(fis, 'WMH', 'trimf', [0.60 1 1], 'Name', 'wmh');

%% Rules
% Rule format: 
%[intensity mean std skew kurtosis contrast range output weight operator]

% 0 = don't care
% Input MF indexes:
%   low = 1, medium = 2, high = 3
% Output MF indexes:
%   non_wmh = 1, wmh = 2
% Operator:
%   1 = AND, 2 = OR

ruleList = [
    3 3 0 0 0 0 0  2 1 1;  % R1: high intensity AND high local mean -> WMH
    3 0 0 0 0 3 0  2 1 1;  % R2: high intensity AND high local contrast -> WMH
    3 0 2 0 0 0 0  2 1 1;  % R3: high intensity AND medium std -> WMH
    3 0 3 0 0 0 0  2 1 1;  % R4: high intensity AND high std -> WMH
    0 3 0 0 3 0 0  2 1 1;  % R5: high local mean AND high kurtosis -> WMH
    3 0 0 3 0 0 0  2 1 1;  % R6: high intensity AND high skewness -> WMH
    2 3 3 0 0 0 0  2 1 1;  % R7: medium intensity AND high mean AND high std -> WMH
    3 0 0 0 0 0 3  2 1 1;  % R8: high intensity AND high local range -> WMH
    2 3 0 0 0 0 3  2 1 1;  % R9: medium intensity AND high mean AND high local range -> WMH
    1 0 0 0 0 0 0  1 1 1;  % R10: low intensity -> non-WMH
    0 1 0 0 0 0 0  1 1 1;  % R11: low local mean -> non-WMH
    0 0 1 0 0 1 1  1 1 1;  % R12: low std AND low contrast AND low range -> non-WMH
];

fis = addRule(fis, ruleList);

% visualization
if showPlots
    figure('Name', 'Intensity MFs')
    plotmf(fis, 'input', 1, 1000);
    title('Pixel Intensity Membership Functions')

    figure('Name', 'Local Range MFs')
    plotmf(fis, 'input', 7, 1000);
    title('Local Range Membership Functions')

    figure('Name', 'WMH Output MFs')
    plotmf(fis, 'output', 1, 1000);
    title('WMH Output Membership Functions')

end
