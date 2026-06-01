function fis = fis_rules(showPlots)
%% WMH Fuzzy Inference System
% use Mamdani FIS

fis = mamfis('Name', 'WMH_FIS');

% intensity
fis = addInput(fis, [0 1], 'Name', 'intensity');
fis = addMF(fis, 'intensity', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'intensity', 'trimf', [0.35 0.55 0.72], 'Name', 'medium');
fis = addMF(fis, 'intensity', 'trapmf', [0.62 0.72 1 1], 'Name', 'high');

% mean
fis = addInput(fis, [0 1], 'Name', 'mean');
fis = addMF(fis, 'mean', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'mean', 'trimf', [0.35 0.55 0.72], 'Name', 'medium');
fis = addMF(fis, 'mean', 'trapmf', [0.62 0.72 1 1], 'Name', 'high');

% std
fis = addInput(fis, [0 1], 'Name', 'std');
fis = addMF(fis, 'std', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'std', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'std', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% skewness
fis = addInput(fis, [0 1], 'Name', 'skewness');
fis = addMF(fis, 'skewness', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'skewness', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'skewness', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% kurtosis
fis = addInput(fis, [0 1], 'Name', 'kurtosis');
fis = addMF(fis, 'kurtosis', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'kurtosis', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'kurtosis', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% contrast
fis = addInput(fis, [0 1], 'Name', 'contrast');
fis = addMF(fis, 'contrast', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'contrast', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'contrast', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% range
fis = addInput(fis, [0 1], 'Name', 'range');
fis = addMF(fis, 'range', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'range', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'range', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

% spatial coordinate x
fis = addInput(fis, [0 1], 'Name', 'x');
fis = addMF(fis, 'x', 'trapmf', [0 0 0.20 0.40], 'Name', 'left');
fis = addMF(fis, 'x', 'trimf', [0.25 0.50 0.75], 'Name', 'center');
fis = addMF(fis, 'x', 'trapmf', [0.60 0.80 1 1], 'Name', 'right');

% spatial coordinate y
fis = addInput(fis, [0 1], 'Name', 'y');
fis = addMF(fis, 'y', 'trapmf', [0 0 0.20 0.40], 'Name', 'top');
fis = addMF(fis, 'y', 'trimf', [0.25 0.50 0.75], 'Name', 'middle');
fis = addMF(fis, 'y', 'trapmf', [0.60 0.80 1 1], 'Name', 'bottom');

% output
fis = addOutput(fis, [0 1], 'Name', 'WMH');
fis = addMF(fis, 'WMH', 'trimf', [0 0 0.40], 'Name', 'non_wmh');
fis = addMF(fis, 'WMH', 'trimf', [0.60 1 1], 'Name', 'wmh');

% rules
% [intensity mean std skew kurtosis contrast range x y output weight operator]
% low/left/top = 1, medium/center/middle = 2, high/right/bottom = 3
% weight = 1 for all rules
% operator = 1 AND
% output: non_wmh = 1, wmh = 2

ruleList = [
    3 3 0 0 0 0 0 0 0  2 1 1;  % high intensity AND high mean -> WMH
    3 0 0 0 0 3 0 0 0  2 1 1;  % high intensity AND high contrast -> WMH
    3 0 2 0 0 0 0 0 0  2 1 1;  % high intensity AND medium std -> WMH
    3 0 3 0 0 0 0 0 0  2 1 1;  % high intensity AND high std -> WMH
    0 3 0 0 3 0 0 0 0  2 1 1;  % high mean AND high kurtosis -> WMH
    3 0 0 3 0 0 0 0 0  2 1 1;  % high intensity AND high skewness -> WMH
    2 3 3 0 0 0 0 0 0  2 1 1;  % medium intensity AND high mean AND high std -> WMH
    3 0 0 0 0 0 3 0 0  2 1 1;  % high intensity AND high range -> WMH
    2 3 0 0 0 0 3 0 0  2 1 1;  % medium intensity AND high mean AND high range -> WMH
    3 0 0 0 0 3 0 2 2  2 1 1;  % high intensity AND contrast in central area -> WMH
    0 3 0 0 3 0 0 2 2  2 1 1;  % high mean AND kurtosis in central area -> WMH
    1 0 0 0 0 0 0 0 0  1 1 1;  % low intensity -> non WMH
    0 1 0 0 0 0 0 0 0  1 1 1;  % low mean -> non WMH
    0 0 1 0 0 1 1 0 0  1 1 1;  % low std AND low contrast AND low range -> non-WMH
];

fis = addRule(fis, ruleList);

if showPlots
    figure('Name', 'Intensity MFs')
    plotmf(fis, 'input', 1, 1000);
    title('Pixel Intensity Membership Functions')

    figure('Name', 'Range MFs')
    plotmf(fis, 'input', 7, 1000);
    title('Range Membership Functions')

    figure('Name', 'Spatial Coordinate MFs')
    subplot(1, 2, 1)
    plotmf(fis, 'input', 8, 1000);
    title('X Coordinate Membership Functions')
    subplot(1, 2, 2)
    plotmf(fis, 'input', 9, 1000);
    title('Y Coordinate Membership Functions')

    figure('Name', 'WMH Output MFs')
    plotmf(fis, 'output', 1, 1000);
    title('WMH Output Membership Functions')
end
end
