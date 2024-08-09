function d = distGrid(K,M)
% This computes computes the Euclidean distance between the coordinates of two grids
% Exact WB for grid-structured data
%
% K is s.t K X K is the image grid
% M is the number of images
[X,Y] = meshgrid(1:1/M:K);
ptsK  = [X(:), Y(:)];
[X,Y] = meshgrid(1:1:K);
ptsk  = [X(:), Y(:)]; clear X Y
d     = pdist2(ptsK,ptsk);
return
