function fis = wmh_build_fis()
% Build the WMH Fuzzy Inference System used by the MATLAB and Python code.
% The rules are the same as in wmh_fis_matlab_rules.m.

fis = mamfis('Name', 'WMH_FIS');

%% Inputs
fis = addInput(fis, [0 1], 'Name', 'intensity');
fis = addMF(fis, 'intensity', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'intensity', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'intensity', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

fis = addInput(fis, [0 1], 'Name', 'local_mean');
fis = addMF(fis, 'local_mean', 'trapmf', [0 0 0.20 0.45], 'Name', 'low');
fis = addMF(fis, 'local_mean', 'trimf', [0.25 0.50 0.75], 'Name', 'medium');
fis = addMF(fis, 'local_mean', 'trapmf', [0.55 0.75 1 1], 'Name', 'high');

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

%% Output
fis = addOutput(fis, [0 1], 'Name', 'WMH');
fis = addMF(fis, 'WMH', 'trimf', [0 0 0.40], 'Name', 'non_wmh');
fis = addMF(fis, 'WMH', 'trimf', [0.60 1 1], 'Name', 'wmh');

%% Rules
ruleList = [
    3 3 0 0 0 0 0  2 1 1;  % R1
    3 0 0 0 0 3 0  2 1 1;  % R2
    3 0 2 0 0 0 0  2 1 1;  % R3
    3 0 3 0 0 0 0  2 1 1;  % R4
    0 3 0 0 3 0 0  2 1 1;  % R5
    3 0 0 3 0 0 0  2 1 1;  % R6
    2 3 3 0 0 0 0  2 1 1;  % R7
    3 0 0 0 0 0 3  2 1 1;  % R8
    2 3 0 0 0 0 3  2 1 1;  % R9
    1 0 0 0 0 0 0  1 1 1;  % R10
    0 1 0 0 0 0 0  1 1 1;  % R11
    0 0 1 0 0 1 1  1 1 1;  % R12
];

fis = addRule(fis, ruleList);

end
