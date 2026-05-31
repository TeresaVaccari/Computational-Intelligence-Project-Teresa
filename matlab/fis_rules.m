function fis = fis_rules(showPlots)
%% WMH Fuzzy Inference System
% Mamdani FIS for the nine normalized WMH features exported to MATLAB:
% intensity, local mean, local std, skewness, kurtosis, local contrast,
% local range, x coordinate, and y coordinate.


fis = mamfis('Name', 'WMH_FIS');

%% inputs
fis = addInput(fis, [0 1], 'Name', 'intensity');
fis = addMF(fis, 'intensity', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'intensity', 'trimf', [0.35 0.55 0.72], 'Name', 'medium');
fis = addMF(fis, 'intensity', 'trapmf', [0.62 0.72 1 1], 'Name', 'high');

fis = addInput(fis, [0 1], 'Name', 'local_mean');
fis = addMF(fis, 'local_mean', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_mean', 'trimf', [0.35 0.55 0.72], 'Name', 'medium');
fis = addMF(fis, 'local_mean', 'trapmf', [0.62 0.72 1 1], 'Name', 'high');

fis = addInput(fis, [0 1], 'Name', 'local_std');
fis = addMF(fis, 'local_std', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_std', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'local_std', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

fis = addInput(fis, [0 1], 'Name', 'skewness');
fis = addMF(fis, 'skewness', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'skewness', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'skewness', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

fis = addInput(fis, [0 1], 'Name', 'kurtosis');
fis = addMF(fis, 'kurtosis', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'kurtosis', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'kurtosis', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

fis = addInput(fis, [0 1], 'Name', 'local_contrast');
fis = addMF(fis, 'local_contrast', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_contrast', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'local_contrast', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

fis = addInput(fis, [0 1], 'Name', 'local_range');
fis = addMF(fis, 'local_range', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_range', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'local_range', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

fis = addInput(fis, [0 1], 'Name', 'x');
fis = addMF(fis, 'x', 'trapmf', [0 0 0.20 0.40], 'Name', 'left');
fis = addMF(fis, 'x', 'trimf', [0.25 0.50 0.75], 'Name', 'center');
fis = addMF(fis, 'x', 'trapmf', [0.60 0.80 1 1], 'Name', 'right');

fis = addInput(fis, [0 1], 'Name', 'y');
fis = addMF(fis, 'y', 'trapmf', [0 0 0.20 0.40], 'Name', 'top');
fis = addMF(fis, 'y', 'trimf', [0.25 0.50 0.75], 'Name', 'middle');
fis = addMF(fis, 'y', 'trapmf', [0.60 0.80 1 1], 'Name', 'bottom');

%% output
fis = addOutput(fis, [0 1], 'Name', 'WMH');
fis = addMF(fis, 'WMH', 'trimf', [0 0 0.40], 'Name', 'non_wmh');
fis = addMF(fis, 'WMH', 'trimf', [0.60 1 1], 'Name', 'wmh');

%% rules
% Rule format:
% [intensity mean std skew kurtosis contrast range x y output weight operator]
% Feature MF indexes: low/left/top = 1, medium/center/middle = 2,
% high/right/bottom = 3. Output: non_wmh = 1, wmh = 2.

ruleList = [
    3 3 0 0 0 0 0 0 0  2 1 1;
    3 0 0 0 0 3 0 0 0  2 1 1;
    3 0 2 0 0 0 0 0 0  2 1 1;
    3 0 3 0 0 0 0 0 0  2 1 1;
    0 3 0 0 3 0 0 0 0  2 1 1;
    3 0 0 3 0 0 0 0 0  2 1 1;
    2 3 3 0 0 0 0 0 0  2 1 1;
    3 0 0 0 0 0 3 0 0  2 1 1;
    2 3 0 0 0 0 3 0 0  2 1 1;
    3 0 0 0 0 3 0 2 2  2 1 1;
    0 3 0 0 3 0 0 2 2  2 1 1;
    1 0 0 0 0 0 0 0 0  1 1 1;
    0 1 0 0 0 0 0 0 0  1 1 1;
    0 0 1 0 0 1 1 0 0  1 1 1;
];

fis = addRule(fis, ruleList);

if showPlots
    figure('Name', 'Intensity MFs')
    plotmf(fis, 'input', 1, 1000);
    title('Pixel Intensity Membership Functions')

    figure('Name', 'Local Range MFs')
    plotmf(fis, 'input', 7, 1000);
    title('Local Range Membership Functions')

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
