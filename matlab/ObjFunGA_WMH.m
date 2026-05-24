function error = ObjFunGA_WMH(chromosome, fis_base, param)
% Function for GA rule-weight tuning.
% MATLAB ga minimizes by default.

for r = 1:length(chromosome)
    fis_base.Rules(r).Weight = chromosome(r);
end

featureMatrices = param.featureMatrices;
manualMasks = param.manualMasks;

errorValues = zeros(1, numel(featureMatrices));

for i = 1:numel(featureMatrices)

    probabilities = evalfis(fis_base, featureMatrices{i});
    target = double(manualMasks{i}(:));
    errorValues(i) = mean((probabilities(:) - target(:)).^2);
end

error = mean(errorValues);
end
