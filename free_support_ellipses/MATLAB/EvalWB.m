function EvalWB
clc
close all
M = 10;
K = 60;
Kn= M*(K-1) + 1;
R = Kn*Kn;

load images.mat
barycenter = double(barycenter');
barycenter = barycenter./sum(barycenter);
data = barycenter;
clear barycenter
for m=1:M
    im = reshape(data(:,m),K,K);
    subplot(2,5,m)
    imagesc(1-im)
    colormap hot
end
load res_altschuler.mat;
pex = Altschuler';

clear Altschuler

figure
imagesc(reshape(1-pex,Kn,Kn));
colormap hot
title('Exact solution')

%load MAM_results.mat;
%p = barycenter';
%clear barycenter
load 6300.mat %3020.mat %MAM_results.mat;
I=find(p<1e-6);
p(I)=0; p= p/sum(p);

figure
imagesc(reshape(1-p,Kn,Kn));
colormap hot
title('MAM solution')
pause(0.01)

disp('Computing the distance matrix...');
D = (distGrid(K,M).^2)/3600;
%D = DistDaniel(R,K*K).^2;

disp('Computing the WB distance for the exact barycenter ...');
I = pex>1e-8;
pex = pex(I);
D   = D(I,:); 
Fex = 0;
for m=1:M
    fprintf('m=%5.0f \n',m);
    im = data(:,m);
    J  = im>1e-8;
    im = im(J);
    d  = D(:,J);
    [~,val] = OT(d,pex,im);
    Fex = Fex + val;
end
Fex

D = (distGrid(K,M).^2)/3600;
%D = DistDaniel(R,K*K).^2;

disp('Computing the WB distance for the MAM barycenter ...');
I = p>1e-8;
p = p(I);
D   = D(I,:); 
F = 0;
for m=1:M
    fprintf('m=%5.0f \n',m);
    im = data(:,m);
    J  = im>1e-8;
    im = im(J);
    d  = D(:,J);
    [~,val] = OT(d,p,im);
    F = F + val;
end
F



return