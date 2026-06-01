% ObjFunGA_WMH function
% assign weights of chromosome to fuzzy inference system

function [ f ] = ObjFunGA_WMH(chromosome, fis_base, param)
% chromosome is vector with fuzzy rules weights
% fis_base is the fuzzy inference system 
% param is the structure with features 

% chromosome contains one weight for each fuzzy rule.
for r = 1:length(chromosome)
    fis_base.Rules(r).Weight = chromosome(r);
end

% create input matrix for train pixel
idx = param.pixel_idx;

input_pixels = [ % each row is a pixel, each column is a feature
    param.intensity(idx), ...
    param.mean(idx), ...
    param.std(idx), ...
    param.skew(idx), ...
    param.kurtosis(idx), ...
    param.contrast(idx), ...
    param.range(idx), ...
    param.x(idx), ...
    param.y(idx)
];

% evaluate fis
fis_scores = evalfis(fis_base, input_pixels);
dice_scores = zeros(size(param.thresholds));

for t = 1:length(param.thresholds)
    prediction = fis_scores >= param.thresholds(t);
    denom = sum(prediction(:)) + sum(param.gold(idx));

    if denom == 0
        dice_scores(t) = 1;
    else
        dice_scores(t) = 2 * sum(prediction(:) & param.gold(idx)) / denom;
    end
end

% GA minimize error, hence maximize dice
f = 1 - max(dice_scores);

end
