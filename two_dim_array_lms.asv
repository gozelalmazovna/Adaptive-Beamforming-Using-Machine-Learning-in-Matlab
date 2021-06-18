function [weighty,weightz,AF] = two_dim_array_lms(My,Nz,dy,dz,lam,theta_desired,theta_interfered)
%Two dimensional array on yz plane
%by Gozel Murrukova 
%My,Nz: number of antennas
%dy,dz: distance between antennas
%theta_desired: desired angle
%theta_interfered: interference angle

weighty=lms_func(My,dy,lam,theta_desired,theta_interfered);
weightz=lms_func(Nz,dz,lam,theta_desired,theta_interfered);

j=sqrt(-1);
M=361; %Angle resolution
k=2*pi; %wavenumber
theta=linspace(0,pi,M);
phi=linspace(-pi/2,pi/2,M);
[THETA,PHI]=meshgrid(theta,phi);
deltay=0; %Steering angle in phi
deltaz=0; %Steering angle in theta
%Array factor
psiy=(-k*dy*sin(THETA).*sin(PHI))+deltay;
psiz=(-k*dz*cos(THETA))+ deltaz;
AFy=0;
AFz=0;
for m=1:My
    AFy= AFy+ weighty(m)'.* exp(j*(m-1)*psiy);
end
for n=1:Nz
    AFz=AFz+ weightz(n)'.*exp(j*(n-1)*psiz);
end
AF=AFy.*AFz;
end


