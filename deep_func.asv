function [w] = deep_func(N,d,lam,thetaS,thetaI)
%Calculates weights using depp learning algorithm
%by Gozel Murrukova 
%   Inputs: N - number of antennas,
%           d - distance between antennas,
%           lambda - lambda value,
%           theta_desired - desired angle,
%           theta_interference - interference angle
%   Outputs: weights for calculation of AF
data =  readmatrix("my_data.csv");
x = data(1:3700,1:5);
y = data(1:3700,6:end);

y(isnan(y)) = 0;
Y_norm = (y - min(y)) ./ ( max(y) - min(y));
x(isnan(x)) = 0;
X_norm = (x - repmat(mean(x),size(x, 1), 1))./repmat(std(x),size(x, 1), 1);

Xtrain = (X_norm(1:3000,:))';
Ytrain = abs(Y_norm(1:3000,:))';

Xvalidation = (X_norm(3001:3700,:))';
Yvalidation = abs(Y_norm(3001:3700,:))';

inputSize = 5;
numResponses = 20;

numHiddenUnits = 100;
layers = [ sequenceInputLayer(inputSize)
    batchNormalizationLayer
     lstmLayer(numHiddenUnits)
     batchNormalizationLayer
     fullyConnectedLayer(numResponses)
     regressionLayer];

opts = trainingOptions('adam',...
       'MaxEpochs',40, ...
       'LearnRateSchedule','piecewise', ...
       'LearnRateDropPeriod',30, ...
       'LearnRateDropFactor',0.2, ...
	   'GradientThreshold',0.01, ...
	   'InitialLearnRate',0.01, ...
        'ValidationData',{Xvalidation,Yvalidation}, ...
        'Plots','training-progress');

net = trainNetwork(Xtrain,Ytrain,layers,opts);

a = [N,d,lam,thetaS,thetaI];
b = a - repmat(mean(a),size(a, 1), 1)./repmat(std(a),size(a, 1), 1);
ypred =(predict(net,(b)'));
w = ypred(1:a(1),:)';

end









