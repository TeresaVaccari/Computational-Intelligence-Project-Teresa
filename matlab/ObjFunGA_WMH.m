% funzione ObjFunGA_WMH
% assegna weights del cromosoma al fuzzy inference system

function [ f ] = ObjFunGA_WMH(chromosome, fis_base, param)
% chromosome è il vettore con i pesi dlle regole fuzzy
% fis_base è il fis creato con fis_rules.m
% param è structure con features 

% The chromosome contains one weight for each fuzzy rule.
for r = 1:length(chromosome)
    fis_base.Rules(r).Weight = chromosome(r);
end

% crea matrice di input per i pixel di training
idx = param.pixel_idx;

input_pixels = [ % ogni row è un pixel, ogni colonna una feature
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

% valuta il fis
probabilities = evalfis(fis_base, input_pixels);
dice_scores = zeros(size(param.thresholds));

for t = 1:length(param.thresholds)
    prediction = probabilities >= param.thresholds(t);
    denom = sum(prediction(:)) + sum(param.gold(idx));

    if denom == 0
        dice_scores(t) = 1;
    else
        dice_scores(t) = 2 * sum(prediction(:) & param.gold(idx)) / denom;
    end
end

% il GA minimizza, quindi minimizziamo 1 - Dice
f = 1 - max(dice_scores);

end
