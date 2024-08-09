function launchMAM
clc
close all
gurobigo
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
    im = reshape(data(:,m)',K,K);
    subplot(2,5,m)
    imagesc(1-im)
    colormap hot
end



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
D = (distGrid(K,M)).^2;
d = cell(M,1);
q = cell(M,1);
S = zeros(1,M);
for m=1:M
    im = data(:,m);
    J  = im>0;
    im = im(J);
    im = im/sum(im);
    S(m) = sum(J);
    q{m} = im;
    d{m}  = D(:,J)/3600;
end
[p,val,cpu] = MAM(d,q,M,R,S,1);