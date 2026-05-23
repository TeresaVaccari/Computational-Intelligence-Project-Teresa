% function for GA rule-weight tuning.
%
% MATLAB ga minimizza di default 
% quindi per massimizzare il Dice medio, definiamo l'errore come 1 - meanDice, 
% in modo che minimizzando l'errore massimizziamo il Dice medio.
%
%   error = 1 - meanDice
%
% The chromosome is ruleWeights:
%   [w1, w2, ..., w12]



function error = ObjFunGA_WMH(ruleWeights, fis, featureMatrices, manualMasks, imageShapes, threshold)

% input: ruleWeights array of weights for the fuzzy rules, values between 0 and 1
% fis fuzzy inference system, created by wmh_fis_rules.m
% featureMatrices cell array of preprocessed feature matrices for each slice (pixel number x feature number)
% manualMasks are the true masks 
% imageShapes reconstruct the score from feature space to image space

fisWeighted = set_rule_weights(fis, ruleWeights); % applies the weights to the fuzzy rules
diceValues = zeros(1, numel(featureMatrices)); % tries all slices

for i = 1:numel(featureMatrices) % create vector to save Dice of each slice
    
    % apply FIS:
    % input featureMatrices{i} 
    % output score, a fuzzy valu btw 0 and 1 for each pixel
    score = evalfis(fisWeighted, featureMatrices{i}); 
    
    % apply threshold to get binary prediction, reshape to image space
    prediction = reshape(score >= threshold, imageShapes{i}(1), imageShapes{i}(2));
    
    % compare with true mask
    target = manualMasks{i};
    diceValues(i) = dice_score(prediction, target); % compute Dice for each slice
end

meanDice = mean(diceValues); % compute mean Dice across slices
error = 1 - meanDice; % return error

end


function fisWeighted = set_rule_weights(fis, ruleWeights)
fisWeighted = fis;
for i = 1:numel(ruleWeights)
    fisWeighted.Rules(i).Weight = ruleWeights(i);
end
end

% compute Dice = 2 * intersection / (prediction + target)
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
