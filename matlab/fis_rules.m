function fis = fis_rules()
% WMH Fuzzy Inference System

fis = mamfis('Name', 'WMH_FIS');

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

fis = addOutput(fis, [0 1], 'Name', 'WMH');
fis = addMF(fis, 'WMH', 'trimf', [0 0 0.40], 'Name', 'non_wmh');
fis = addMF(fis, 'WMH', 'trimf', [0.60 1 1], 'Name', 'wmh');

% [intensity mean std skew kurtosis contrast range output weight operator]
ruleList = [
    3 3 0 0 0 0 0  2 1 1;
    3 0 0 0 0 3 0  2 1 1;
    3 0 2 0 0 0 0  2 1 1;
    3 0 3 0 0 0 0  2 1 1;
    0 3 0 0 3 0 0  2 1 1;
    3 0 0 3 0 0 0  2 1 1;
    2 3 3 0 0 0 0  2 1 1;
    3 0 0 0 0 0 3  2 1 1;
    2 3 0 0 0 0 3  2 1 1;
    1 0 0 0 0 0 0  1 1 1;
    0 1 0 0 0 0 0  1 1 1;
    0 0 1 0 0 1 1  1 1 1;
];
fis = addRule(fis, ruleList);
end
