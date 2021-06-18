function [w] = ann(N,d,lam,thetaS,thetaI)
%Artificial neural network algorithm to calculate weights
%by Gozel Murrukova 
%   Inputs: N - number of antennas,
%           d - distance between antennas,
%           lambda - lambda value,
%           theta_desired - desired angle,
%           theta_interference - interference angle
%   Outputs: weights for calculation of AF
data = readmatrix("my_data.csv");
x = data(1:3700,1:5);
y = data(1:3700,6:25);


y(isnan(y)) = 0;
y2 = (y - min(y)) ./ ( max(y) - min(y));
x(isnan(x)) = 0;
X_norm = (x - repmat(mean(x),size(x, 1), 1))./repmat(std(x),size(x, 1), 1);

xt = (X_norm)';
yt = abs(y2');

for j = 1:2
    hiddenLayerSize = [7 7];
    net = fitnet(hiddenLayerSize);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 30/100;
    net.divideParam.testRatio = 0/100;
    [net,tr] = train(net,xt,yt);
end
a = [N,d,lam,thetaS,thetaI];
b = a - repmat(mean(a),size(a, 1), 1)./repmat(std(a),size(a, 1), 1);
ypred = (net((b)'));
w = ypred(1:a(1),:)';

end
