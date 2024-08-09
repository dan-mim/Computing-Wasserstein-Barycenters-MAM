function main
close all
clc

M = 10;  % number of images
figure
Q = zeros(60*60,1); 
K = 60;

%--------------------------------------------------------------------------
%                  Read the data and plot the images
%--------------------------------------------------------------------------
fprintf(1,'Reading data...\n');
% for i=1:M
%     d = load(strcat(num2str(i),'.txt'));
%     d(:,[1 2]) = round(K*d(:,[1 2]));
%     im= zeros(K,K);
%     n = size(d,1);
%     for j = 1:n
%         ii=d(j,1);
%         jj=d(j,2);
%         jj=max(jj,1);
%         im(ii,jj) = d(j,3);
%     end
%     Q(:,i) = reshape(im,K*K,1);
%     subplot(2,5,i)
%     imagesc(1-im)
%     colormap hot
% end
load images.mat
barycenter = double(barycenter');
barycenter = barycenter./sum(barycenter);
Q = barycenter;
clear barycenter
for m=1:M
    im = reshape(Q(:,m),K,K);
    subplot(2,5,m)
    imagesc(1-im)
    colormap hot
end
clear im
%--------------------------------------------------------------------------
%          Compute the distance
%--------------------------------------------------------------------------
fprintf(1,'Computing distance...\n');

Kn= M*(K-1) + 1;
R = Kn*Kn;
% D = DistDaniel(R,K*K).^2;
% mx= 1;
D = (distGrid(K,M).^2)/(K*K);
mx = 1;
%--------------------------------------------------------------------------
%        Arrange the data for MAM
%--------------------------------------------------------------------------
fprintf(1,'Arranging data...\n');
S = [];
q = cell(M,1);
d = cell(M,1);
for m=1:M
    I    = Q(:,m)>1e-15;
    S    = [S, sum(I)];
    d{m} = D(:,I);
    q{m} = Q(I,m)';
    q{m} = q{m}/sum(q{m});
end
clear D Q aux
% fprintf(1,'Gurobi...\n');
% [pex,Fex,cpu]=LP_WB(d,q,M,R,S);
% Fex = mx*Fex
%--------------------------------------------------------------------------
%         Go MAM go!
%--------------------------------------------------------------------------
figure
fprintf(1,'MAM is MAM!\n');
pause(0.01)
[p,~,cpu] = MAM(d,q,M,R,S,mx);
method    = 'MAM : ';
%--------------------------------------------------------------------------
%       Plot the computed barycenter
%--------------------------------------------------------------------------
p = reshape(p,Kn,Kn);
imagesc(1-p);
colormap hot; 
title(strcat(method,num2str(cpu),' s'));
disp('Eat my shit!');
return


