function [w] = adagrad(N,d,lam,thetaS,thetaI)
%Adagrad is gradient stochastic method for calculation of weights
%by Gozel Murrukova 
%   Inputs: N - number of antennas,
%           d - distance between antennas,
%           lambda - lambda value,
%           theta_desired - desired angle,
%           theta_interference - interference angle
%   Outputs: weights for calculation of AF

T=1E-3;
t=(1:100)*T/100;
it=1:100;
S=cos(2*pi*t/T);
thetaS = thetaS*pi/180;            
I = randn(1,100);  
thetaI = thetaI*pi/180;                  
i=1:N;
vS=exp(1j*(i-1)*2*pi*d*sin(thetaS)).';
vI=exp(1j*(i-1)*2*pi*d*sin(thetaI)).';
w = zeros(N,1);     
mu = 0.01; %Fixed step size
wi=zeros(N,max(it));
gx_v = 0;

for n = 1:length(S)
    x = S(n)*vS + I(n)*vI;
    y=w'*x;
    e = conj(S(n)) - y;      
    gx_v = 0.9*gx_v + 0.1*(conj(e)*x).^2;
    w=w+ mu*inv(sqrt(diag(gx_v)))*conj(e)*x;
    wi(:,n)=w;
end

w = (w./w(1));    % normalize results to first weight
end

