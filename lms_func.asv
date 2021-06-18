function [w] = lms_func(N,d,lam,thetaS,thetaI)
%Calculate weights using LMS algorithm which then are plotted in GUI
%by Gozel Murrukova 
%   Inputs: N - number of antennas,
%           d - distance between antennas,
%           lambda - lambda value,
%           theta_desired - desired angle,
%           theta_interference - interference angle
%   Outputs: weights for calculation of AF
j = sqrt(-1);
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
X=(vS+vI);
Rx=X*X';
mu=1/(4*real(trace(Rx)));
wi=zeros(N,max(it));
for n = 1:length(S)
    x = S(n)*vS + I(n)*vI;
    y=w'*x;
    e = conj(S(n)) - y;     
    w=w+mu*conj(e)*x;
    wi(:,n)=w;
end

w = (w./w(1));    

end







